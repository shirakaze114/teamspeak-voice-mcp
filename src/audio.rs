//! Audio helpers: WAV loading, resampling and Opus framing for TeamSpeak.

use std::path::Path;

use anyhow::{Context, Result, bail};
use audiopus::coder::Encoder;
use audiopus::{Application, Bitrate, Channels, SampleRate};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

/// TeamSpeak's Opus sample rate.
pub const TS_RATE: u32 = 48_000;
/// 20 ms at 48 kHz.
pub const TS_FRAME: usize = 960;
/// Sample rate fed to ASR.
pub const ASR_RATE: u32 = 16_000;

const MAX_OPUS_PACKET: usize = 1275;

/// Decode any supported audio file/bytes (wav, mp3, flac, ogg/vorbis) to mono f32.
pub fn decode_mono(bytes: Vec<u8>) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let mss = MediaSourceStream::new(Box::new(std::io::Cursor::new(bytes)), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(&Hint::new(), mss, FormatOptions::default(), MetadataOptions::default())
        .context("unrecognised audio format")?;
    let track = format.default_track(TrackType::Audio).context("no audio track")?;
    let params = track.codec_params.as_ref().and_then(|p| p.audio()).context("no audio params")?;
    let mut decoder =
        symphonia::default::get_codecs().make_audio_decoder(params, &AudioDecoderOptions::default())?;
    let track_id = track.id;
    let mut mono = Vec::new();
    let mut rate = 0;
    let mut buf: Vec<f32> = Vec::new();
    while let Some(packet) = format.next_packet()? {
        if packet.track_id != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };
        rate = decoded.spec().rate();
        let ch = decoded.spec().channels().count().max(1);
        buf.resize(decoded.samples_interleaved(), 0.0);
        decoded.copy_to_slice_interleaved(&mut buf);
        mono.extend(buf.chunks_exact(ch).map(|c| c.iter().sum::<f32>() / ch as f32));
    }
    if rate == 0 {
        bail!("audio contained no samples");
    }
    Ok((mono, rate))
}

/// Decode a file and resample to TeamSpeak's 48 kHz.
pub fn load_for_playback(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let (mono, rate) = decode_mono(bytes)?;
    resample_clip(&mono, rate, TS_RATE)
}

/// Little-endian s16 PCM bytes.
pub fn to_s16le(pcm: &[f32]) -> Vec<u8> {
    pcm.iter().flat_map(|s| ((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes()).collect()
}

/// s16le bytes to f32.
pub fn from_s16le(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect()
}

/// Encode mono PCM as an in-memory 16-bit WAV.
pub fn wav_bytes(pcm: &[f32], rate: u32) -> Result<Vec<u8>> {
    let mut cur = std::io::Cursor::new(Vec::new());
    write_wav(&mut cur, pcm, rate)?;
    Ok(cur.into_inner())
}

/// Resample a whole mono clip.
pub fn resample_clip(input: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to || input.is_empty() {
        return Ok(input.to_vec());
    }
    let mut rs = Fft::<f32>::new(from as usize, to as usize, 1024, 1, FixedSync::Input)?;
    let adapter = InterleavedSlice::new(input, 1, input.len())?;
    let out = rs.process_all(&adapter, input.len(), None)?;
    Ok(out.take_data())
}

/// Streaming 48 kHz → 16 kHz mono resampler for per-user tracks.
pub struct Downsampler {
    rs: Fft<f32>,
    pending: Vec<f32>,
    out_buf: Vec<f32>,
}

impl Downsampler {
    pub fn new() -> Result<Self> {
        let rs = Fft::<f32>::new(
            TS_RATE as usize,
            ASR_RATE as usize,
            TS_FRAME,
            1,
            FixedSync::Input,
        )?;
        let out_len = rs.output_frames_max();
        Ok(Self { rs, pending: Vec::with_capacity(TS_FRAME * 2), out_buf: vec![0.0; out_len] })
    }

    /// Push 48 kHz samples, append resampled 16 kHz samples to `out`.
    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<()> {
        self.pending.extend_from_slice(input);
        while self.pending.len() >= TS_FRAME {
            let n = self.process_chunk()?;
            out.extend_from_slice(&self.out_buf[..n]);
            self.pending.drain(..TS_FRAME);
        }
        Ok(())
    }

    /// Flush the remaining samples (zero padded) and reset state.
    pub fn flush(&mut self, out: &mut Vec<f32>) -> Result<()> {
        if !self.pending.is_empty() {
            let real = self.pending.len();
            self.pending.resize(TS_FRAME, 0.0);
            let n = self.process_chunk()?;
            let keep = (real * ASR_RATE as usize).div_ceil(TS_RATE as usize).min(n);
            out.extend_from_slice(&self.out_buf[..keep]);
            self.pending.clear();
        }
        self.rs.reset();
        Ok(())
    }

    fn process_chunk(&mut self) -> Result<usize> {
        let input = InterleavedSlice::new(&self.pending[..TS_FRAME], 1, TS_FRAME)?;
        let frames = self.out_buf.len();
        let mut output = InterleavedSlice::new_mut(&mut self.out_buf, 1, frames)?;
        let (_, n) = self.rs.process_into_buffer(&input, &mut output, None)?;
        Ok(n)
    }
}

/// Encodes 48 kHz mono PCM into 20 ms Opus frames.
pub struct OpusFramer {
    enc: Encoder,
    pending: Vec<f32>,
    out: [u8; MAX_OPUS_PACKET],
}

impl OpusFramer {
    pub fn new() -> Result<Self> {
        let mut enc = Encoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip)?;
        enc.set_bitrate(Bitrate::BitsPerSecond(48_000))?;
        Ok(Self { enc, pending: Vec::new(), out: [0; MAX_OPUS_PACKET] })
    }

    pub fn push(&mut self, pcm: &[f32], mut sink: impl FnMut(Vec<u8>) -> Result<()>) -> Result<()> {
        self.pending.extend_from_slice(pcm);
        while self.pending.len() >= TS_FRAME {
            let n = self.enc.encode_float(&self.pending[..TS_FRAME], &mut self.out)?;
            sink(self.out[..n].to_vec())?;
            self.pending.drain(..TS_FRAME);
        }
        Ok(())
    }

    pub fn finish(&mut self, sink: impl FnMut(Vec<u8>) -> Result<()>) -> Result<()> {
        if !self.pending.is_empty() {
            let pad = TS_FRAME - self.pending.len();
            self.push(&vec![0.0; pad], sink)?;
        }
        Ok(())
    }
}

/// Write mono f32 PCM as a 16-bit WAV file.
pub fn write_wav_mono(path: &Path, pcm: &[f32], rate: u32) -> Result<()> {
    write_wav(std::io::BufWriter::new(std::fs::File::create(path)?), pcm, rate)
}

fn write_wav<W: std::io::Write + std::io::Seek>(w: W, pcm: &[f32], rate: u32) -> Result<()> {
    let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::new(w, spec)?;
    for &s in pcm {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsampler_length() {
        let mut d = Downsampler::new().unwrap();
        let mut out = Vec::new();
        let input: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        for chunk in input.chunks(700) {
            d.push(chunk, &mut out).unwrap();
        }
        d.flush(&mut out).unwrap();
        assert!((out.len() as i64 - 16_000).abs() <= 320, "got {}", out.len());
    }

    #[test]
    fn opus_framer_frames() {
        let mut f = OpusFramer::new().unwrap();
        let mut frames = 0;
        let pcm = vec![0.1f32; TS_FRAME * 3 + 10];
        f.push(&pcm, |_| {
            frames += 1;
            Ok(())
        })
        .unwrap();
        f.finish(|_| {
            frames += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(frames, 4);
    }

    #[test]
    fn wav_roundtrip_via_symphonia() {
        let pcm: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let (mono, rate) = decode_mono(wav_bytes(&pcm, 16_000).unwrap()).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(mono.len(), pcm.len());
    }

    #[test]
    fn resample_clip_ratio() {
        let input = vec![0.0f32; 22_050];
        let out = resample_clip(&input, 22_050, 48_000).unwrap();
        assert!((out.len() as i64 - 48_000).abs() <= 64, "got {}", out.len());
    }
}
