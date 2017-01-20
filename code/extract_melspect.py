#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reads a .wav or other input file and writes out one or more mel or log spectrograms.

For usage information, call with --help.

Author: Jan Schlüter and Thomas Grill
"""

from optparse import OptionParser
from itertools import izip, chain
import numpy as np
import wave

from filterbank import FilterBank

def opts_parser():
	usage =\
"""\r%prog: Reads a .wav or other input file and writes out one or more
mel or log spectrograms.

Usage: %prog [OPTIONS] INFILE OUTFILE
  INFILE: .wav file (16 bit, mono), .raw file (32 bit floats, mono, no header)
      or any file supported by ffmpeg
  OUTFILE: .npz/.h5 output file, features will be called \'melspect<framelen>\',
      or .npy output file, valid only if a single framelength was requested
"""
	parser = OptionParser(usage=usage)
	parser.add_option('-r', '--sample-rate',
			type='int', default=44100,
			help='Sample rate in Hz (default: %default)')
	parser.add_option('-f', '--frame-rate',
			type='float', default=100,
			help='Spectrogram frame rate in Hz (default: %default)')
	parser.add_option('-l', '--frame-lengths',
			type='str', default='2048',
			help='Comma-separated list of spectrogram frame lengths in samples '
				'(default: %default)')
	parser.add_option('--channels', metavar='TREATMENT',
			type='choice', choices=('mix-before', 'mix-after', 'concat', 'split'), default='mix-before',
			help='What to do with multi-channel (stereo) audio files: '
				'"mix-before": downmix the signal before computing spectra; '
				'"mix-after": compute separate spectra and downmix them (before magnitude scaling); '
				'"concat": compute separate spectra and concatenate them to a 3-tensor; '
				'"split": compute separate spectra and save them as separate matrices (appending ".0", ".1" to the names). '
				'(default: %default)')
	parser.add_option('-o', '--online',
			action='store_true', default=False,
			help='Place each window to the left of its reference sample, i.e., '
				'only use past information for each frame')
	parser.add_option('-t', '--freq-scale',
			type='choice', choices=('mel', 'log', 'linear'), default='mel',
			help='Type of filterbank frequency scale: mel, log, linear (default: %default) '
				'For linear, --bands is ignored.')
	parser.add_option('-b', '--bands',
			type='int', default=80,
			help='Number of filters in filterbank (default: %default)')
	parser.add_option('-m', '--min-freq',
			type='float', default=27.5,
			help='Minimum frequency of lowest mel filter in Hz '
				'(default: %default)')
	parser.add_option('-M', '--max-freq',
			type='float', default='16000',
			help='Maximum frequency of highest mel filter in Hz '
				'(default: %default)')
	parser.add_option('-s', '--mag-scale',
			type='choice', choices=('linear','power','log', 'phon', 'sone'), default='log',
			help='Magnitude scaling (linear, power, log, phon or sone) (default: %default)')
	parser.add_option('--log-stretch',
			type='float', default=1.0,
			help='Stretch factor in logarithmic magnitude scaling: '
				'log(shift + stretch * magnitude); (default: %default)')
	parser.add_option('--log-shift',
			type='float', default=0.0,
			help='Shift in logarithmic magnitude scaling: '
				'log(shift + stretch * magnitude); (default: %default)')
	parser.add_option('--db-max',
			type='float', default=96.,
			help='Full scale audio dB SPL equivalent (default: %default)')
	parser.add_option('--keep-phases',
			action='store_true', default=False,
			help='If given, compute complex spectra including the phases.')
	parser.add_option('--preserve-energy',
			action='store_true', default=False,
			help='If given, preserve the per-bin energy of the transformation '
				'(always true for phon or sone magnitudes).')
	parser.add_option('--featname',
			type='str', default='melspect%(len)s',
			help='Template for the names of the feature matrices in the output '
				'file. If present, the string %(len)s is replaced by the '
				'spectrogram frame length in samples. (default: %default)')
	parser.add_option('--include-times',
			action='store_true', default=False,
			help='If given, the output file will contain a vector "times" of time '
				'stamps the same length of the spectrograms giving the position '
				'of each frame\'s reference sample in seconds.')
	parser.add_option('--times-mode',
			type='choice', choices=('beginnings', 'centers', 'borders', 'borders2'),
			default='borders',
			help='Set mode for output time stamps, '
				'with a choice of beginnings, centers, or borders. '
				'(default: %default)')
	return parser

def read_wave(infile, sample_rate, downmix=True):
	f = wave.open(infile)
	try:
		if (f.getsampwidth() != 2) or (f.getframerate() != sample_rate):
			raise ValueError("Unsupported wave file. Needs 16 bits, %d Hz." % sample_rate)
		num_channels = f.getnchannels()
		num_samples = f.getnframes()
		samples = f.readframes(num_samples)
	finally:
		f.close()
	samples = np.frombuffer(samples, dtype=np.int16)
	samples = (samples / 2.**15).astype(np.float32)
	if not downmix:
		# just split up the interleaved samples into one row per channel
		samples = samples.reshape((num_channels, -1), order='F')
	elif num_channels == 2:
		# explicitly downmix stereo files
		samples = (samples[::2] + samples[1::2]) / 2
	elif num_channels > 2:
		raise ValueError("Unsupported wave file. Needs mono or stereo.")
		# we could do a general downmix from the reshaped samples, but
		# that's quite a bit slower than the explicit addition above.
	return samples

def get_num_channels(infile, cmd='avprobe'):
	import subprocess
	info = subprocess.check_output([cmd, "-v", "quiet", "-show_streams", infile])
	for l in info.split():
		if l.startswith('channels='):
			return int(l[len('channels='):])
	return 0

def read_ffmpeg(infile, sample_rate, downmix=True, cmd='avconv'):
	import subprocess
	call = [cmd, "-v", "quiet", "-y", "-i", infile, "-f", "f32le", "-ar", str(sample_rate), "pipe:1"]
	if downmix:
		call[8:8] = ["-ac", "1"]
	else:
		num_channels = get_num_channels(infile, cmd[:2]+'probe') or 1
	samples = subprocess.check_output(call)
	samples = np.frombuffer(samples, dtype=np.float32)
	if not downmix:
		samples = samples.reshape((num_channels, -1), order='F')
	return samples	


class Phonify:
	"""Convert dB SPL to phon units by applying the Terhardt outer ear transfer function.
	The full scale signal dB SPL equivalent needs to be given on initialization (defaults as 96 dB)"""
	def __init__(self,frqs,dB_max=96.,bias=1.e-8,clip=True):
		self.frqs = frqs
		self.todB = lambda x: self.lintodB(x,bias)
		self.corr = self.terhardt_dB(frqs)-self.terhardt_dB(1000)+dB_max
		self.clip = clip

	@staticmethod
	def lintodB(x,bias=1.e-8):
		x = x+bias
		np.log10(x,out=x)
		x *= 20.
		return x

	@staticmethod
	def terhardt_dB(f):
		fk = f/1000.
		return -3.64*fk**-0.8 + 6.5*np.exp(-0.6*(fk-3.3)**2) - 1.e-3*(fk)**4

	def __call__(self,frames,out=None):
		frames = np.asarray(frames)
		spec = self.todB(frames)

		if out is None:
			out = np.empty_like(spec)
		out[:] = spec
		out += self.corr  # apply out-ear transfer function (equal-loudness contours)
		if self.clip:
			np.maximum(out, 0., out=out)  # clip values below the hearing threshold
		return out # dB SPL

def sonify(phons,out=None):
	"""Convert phon to sone units"""
	phons = np.asarray(phons)
	if out is None:
		out = np.empty_like(phons)
	out[:] = phons
	out -= 40.
	out *= 0.1
	np.power(2.,out,out=out)
	idx = out < 1. # out-of-place
	small = phons[idx]
	small *= (1./40.)
	np.power(small,2.642,out=small)
	out[idx] = small
	return out

def logarithmize(spect, stretch=1.0, shift=0.0):
	"""Returns log(shift + stretch * spect). Works in-place for non-complex
	input, works only on the magnitudes and not in-place for complex input."""
	if np.iscomplexobj(spect):
		phases = np.angle(spect)
		spect = np.abs(spect)
	else:
		phases = None
	if stretch != 1:
		spect *= stretch
	if shift == 0:
		eps = 2.220446049250313e-16  # compatibility with yaafe
		np.maximum(spect, eps, spect)
		np.log(spect, spect)
	elif shift == 1:
		np.log1p(spect, spect)
	else:
		spect += shift
		np.log(spect, spect)
	if phases is not None:
		spect = spect * np.exp(1.j*phases)
	return spect



def filtered_stft(samples, framelen, hopsize, transmat, online=False, keep_phases=False, periodic_window=False, normalize_fft=False):
	if periodic_window:
		window = np.hanning(framelen+1)[1:]
	else:
		window = np.hanning(framelen)

	if normalize_fft:
		window *= 1./np.sqrt(np.mean(window**2))
		window *= 2./framelen

	if samples.ndim == 1:
		zeropad = np.zeros(framelen//2, dtype=samples.dtype)
	else:
		zeropad = np.zeros((samples.shape[0], framelen//2), dtype=samples.dtype)
	if online:
		samples = np.concatenate((zeropad, zeropad, samples))
	else:
		samples = np.concatenate((zeropad, samples, zeropad), axis=samples.ndim-1)

	if isinstance(transmat, slice):
		if not keep_phases:
			def process(x):
				return np.abs(x)[transmat]
		else:
			def process(x):
				return x[transmat]
	else:
		if not keep_phases:
			def process(x):
				return np.dot(np.abs(x), transmat)
		else:
			def process(x):
				m = np.dot(np.abs(x), transmat)
				p = np.angle(np.dot(x, transmat))
				return m * np.exp(1.j * p)
	if samples.ndim == 1:
		spect = np.vstack(process(np.fft.rfft(samples[pos:pos+framelen] * window))
				for pos in xrange(0, len(samples) - framelen, int(hopsize)))
	else:
		spect = np.vstack(chain.from_iterable((process(np.fft.rfft(samp * window))
				for samp in samples[:, pos:pos+framelen])
				for pos in xrange(0, samples.shape[-1] - framelen, int(hopsize))))
		spect = spect.reshape(-1, samples.shape[0], spect.shape[-1])
	return spect

def compute_spect(samples, sample_rate, fps=100, framelens=(2048,),
		freq_scale='mel', downmix=False, online=False, bands=80, min_freq=27.5, max_freq=16000,
		mag_scale=('log', 1.0, 0.0), keep_phases=False, periodic_window=False, preserve_energy=False):
	# apply STFTs and mel bank and logarithmize
	hopsize = sample_rate / fps
	result = list()

	for framelen in framelens:
		if freq_scale == 'linear':
			fft_freqs = np.linspace(0, sample_rate / 2., num=framelen // 2 + 1)
			low, high = np.searchsorted(fft_freqs, [min_freq, max_freq])
			bank = slice(low, high)
		else:
			bank = FilterBank(framelen // 2 + 1, sample_rate, num_filters=bands, min_freq=min_freq, max_freq=max_freq, scale=freq_scale, shape='tri', dtype=np.double, preserve_energy=preserve_energy)
			bank = bank.as_matrix()
		spect = filtered_stft(samples, framelen, hopsize, bank, online=online, keep_phases=keep_phases, periodic_window=periodic_window, normalize_fft=preserve_energy)
		if downmix:
			spect = spect.mean(axis=1)
		if mag_scale[0] == 'log':
			spect = logarithmize(spect, stretch=mag_scale[1], shift=mag_scale[2])
		elif mag_scale[0] == 'power':
			np.square(spect,out=spect)
		elif mag_scale[0] in ('phon','sone'):
			phonify = Phonify(bank.peaks_freq[1:-1],dB_max=mag_scale[1])
			phonify(spect,out=spect)
			if mag_scale[0] == 'sone':
				sonify(spect,out=spect)
		result.append(spect.astype(np.float32 if not keep_phases else np.complex64))
	return result

def extract_melspect(infile, sample_rate, **args):
	# read input samples
	downmix = (args['downmix'] == 'before')
	if infile.endswith('.raw'):
		samples = np.memmap(infile, dtype=np.float32)
	else:
		try:
			samples = read_wave(infile, sample_rate, downmix)
		except (wave.Error, ValueError):
			try:
				samples = read_ffmpeg(infile, sample_rate, downmix)
			except OSError:
				samples = read_ffmpeg(infile, sample_rate, downmix, cmd='ffmpeg')
	# transform signal to spectrum
	args['downmix'] = (args['downmix'] == 'after')
	return compute_spect(samples, sample_rate, **args)

def main():
	# parse command line
	parser = opts_parser()
	options, args = parser.parse_args()
	if len(args) != 2:
		parser.error("missing INFILE or OUTFILE")
	infile, outfile = args
	framelens = map(int, options.frame_lengths.split(','))

	if (len(framelens) > 1) and (outfile.endswith('.npy')):
		parser.error(".npy output not supported for more than one frame length")

	if options.mag_scale == 'linear':
		mag_scale = ('linear',)
	elif options.mag_scale == 'log':
		mag_scale = ('log', options.log_stretch, options.log_shift)
	elif options.mag_scale in ('phon','sone'):
		mag_scale = (options.mag_scale, options.db_max)
		options.preserve_energy = True

	# call extract_melspect()
	if options.channels == 'mix-before':
		downmix = 'before'
	elif options.channels == 'mix-after':
		downmix = 'after'
	else:
		downmix = False
	spects = extract_melspect(infile, options.sample_rate,
			fps=options.frame_rate, framelens=framelens,
			downmix=downmix, online=options.online,
			freq_scale=options.freq_scale, bands=options.bands,
			min_freq=options.min_freq, max_freq=options.max_freq,
			mag_scale=mag_scale, keep_phases=options.keep_phases,
			periodic_window=options.preserve_energy,
			preserve_energy=options.preserve_energy)

	# write to output file
	if outfile.endswith('.npy'):
		np.save(outfile, spects[0])
	else:
		data = [(options.featname % {'len': flen}, spect) for flen, spect in izip(framelens, spects)]
		if options.channels == 'split':
			data = sum(([(n + '.' + str(i), spect[:,i]) for i in xrange(spect.shape[1])]
					for n, spect in data), [])
		if options.include_times:
			dt = 1./options.frame_rate
			times = np.arange(len(spects[0])+1,dtype=np.float32)*dt
			if options.times_mode == 'beginnings':
				times = times[:-1]
			elif options.times_mode == 'centers':
				# shift times to bin centers
				times = times[:-1]+dt/2.
			elif options.times_mode == 'borders':
				pass
			elif options.times_mode == 'borders2':
				# give the left and right border per frame
				times = np.vstack((times[:-1], times[:-1] + float(framelens[0]) / options.sample_rate)).T
			else:
				raise NotImplementedError("Option --times-mode choice '%s' unhandled."%options.times_mode)
			data.append(('times', times))
		if outfile.endswith('.h5'):
			import h5py
			with h5py.File(outfile, 'w') as f:
				for k, v in data:
					f[k] = v
				for k, v in options.__dict__.iteritems():
					f.attrs[k] = v
		else:
			np.savez(outfile, **dict(data))

if __name__=="__main__":
	main()
