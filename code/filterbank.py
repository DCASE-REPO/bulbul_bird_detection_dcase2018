#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for creating and applying a frequency scale (mel or log) filterbank.

Author: Jan Schlüter and Thomas Grill
"""

import numpy as np

def mel_spaced_frequencies(count, min_freq, max_freq):
	"""
	Creates an array of frequencies spaced on a Mel scale.
	@param count: The number of frequencies to return
	@param min_freq: Lowest frequency in Hz
	@param max_freq: Highest frequency in Hz
	@return A vector of `count` frequencies in Hz (including `min_freq` and
		`max_freq`)
	"""
	# minimum, maximum and spacing in mel
	min_mel = 1127 * np.log1p(min_freq / 700.0)
	max_mel = 1127 * np.log1p(max_freq / 700.0)
	spacing = (max_mel - min_mel) / (count - 1)
	# peaks of the filters in mel and in Hz
	peaks_mel = min_mel + np.arange(count) * spacing
	peaks_freq = 700 * (np.exp(peaks_mel / 1127) - 1)
	return peaks_freq

def log_spaced_frequencies(count, min_freq, max_freq):
	"""
	Creates an array of frequencies logarithmically spaced.
	@param count: The number of frequencies to return
	@param min_freq: Lowest frequency in Hz
	@param max_freq: Highest frequency in Hz
	@return A vector of `count` frequencies in Hz (including `min_freq` and
		`max_freq`)
	"""
	# minimum, maximum and spacing in mel
	min_log = np.log(min_freq)
	max_log = np.log(max_freq)
	peaks_freq = np.logspace(min_log,max_log,base=np.e,num=count,endpoint=True)
	return peaks_freq


class FilterBank(object):
	"""
	Encapsulates a frequency scale filterbank. Offers to apply the filterbank to given
	input data, or to return a transformation matrix representing the bank.
	"""

	fscales = {'mel': mel_spaced_frequencies, 'log': log_spaced_frequencies}
	
	def __init__(self, length, sample_rate, num_filters, min_freq=130.0, max_freq=6854.0, norm=True, scale='mel', shape='tri', dtype=np.double, preserve_energy=False):
		"""
		Creates a new mel or log filterbank instance.
		@param length: Length of frames (in samples) the bank is to be
			applied to
		@param sample_rate: Sample rate of input data (used to calculate
			the cutoff frequencies)
		@param num_filters: The number of filters (the number of frequency bands)
		@param min_freq: The low cutoff point of the lowest filter
		@param max_freq: The high cutoff point of the highest filter
		@param norm: Whether to normalize each filter to unit area
		@param scale: mel or log
		@param shape: The filter shape: 'tri' for triangular, 'hann' for hann
		@param dtype: The dtype of the filters
		"""
		# Creates a mel or log filter bank. The mel bank is compatible to what yaafe uses

		self.sample_rate = sample_rate
		self.length = length
		self.num_filters = num_filters
		self.dtype = dtype
		
		# - filter bank peak frequencies
		try:
			fscale = self.fscales[scale]
		except KeyError:
			raise ValueError("scale parameter '%s' not recognized (must be 'mel' or 'log')"%scale)
		self.peaks_freq = fscale(num_filters + 2, min_freq, max_freq)

		nyquist = float(sample_rate)/2.

		# - frequency at each fft bin
		fft_freqs = np.linspace(0,nyquist,num=length)

		if preserve_energy:
			pr_factor = nyquist/length  # bin energy (mean) preservation
#			pr_factor = nyquist/np.sqrt(num_filters*length) # total energy (sum) preservation
		else:
			pr_factor = 1.

		# - prepare list of filters
		self._filters = []

		# - create triangular filters
		for b in xrange(1, num_filters + 1):
			# The triangle starts at the previous filter's peak (peaks_freq[b-1]),
			# has its maximum at peaks_freq[b] and ends at peaks_freq[b+1].
			left, top, right = self.peaks_freq[b-1:b+2]  #b-1, b, b+1
			# Find out where to put the triangle in frequency space
			l, t, r = np.searchsorted(fft_freqs, [left, top, right])
			# Create the filter (equal to yaafe):
			if shape == 'tri':
#				filt = np.bartlett(r-l).astype(dtype)  # alternative (not equal to yaafe)
				filt = np.empty(r-l, dtype=dtype)
				filt[:t-l] = (fft_freqs[l:t] - left)/(top - left)
				filt[t-l:] = (right - fft_freqs[t:r])/(right - top)
				if norm:  # Normalize each filter to unit filter energy.
					filt *= 2. / ((right - left)/pr_factor)
			elif shape == 'hann':
				filt = np.hanning(r-l).astype(dtype)
				if norm:  # Normalize each filter to unit filter energy.
					filt *= 2. / ((right - left)/pr_factor)
			else:
				raise ValueError('Unsupported value for parameter `shape`.')
			# Append to the list of filters
			self._filters.append((l, filt))

	def as_matrix(self,sparse=False):
		"""
		Returns the filterbank as a transformation matrix of shape
		(self.length, self.num_filters). This can be right-multiplied
		to input data to apply the filterbank (inefficiently, however).
		@param sparse: If true, the transformation matrix is returned
			in the CSR format.
			Applying the dot product as mat.T.dot(sig.T) might be faster
			than np.dot(sig,mat) on the dense matrix.
		"""
		mat = np.zeros((self.length, self.num_filters), dtype=self.dtype)
		for b, (l, filt) in enumerate(self._filters):
			mat[l:l+len(filt), b] = filt
		if sparse:
			from scipy.sparse import csr_matrix
			mat = csr_matrix(mat)
		return mat

	def apply(self, data):
		"""
		Applies the filterbank to the given input data. This is meant to be
		more efficient than a dot product with the filter matrix, but it can
		actually be slower (depending on your BLAS implementation).
		@param data: Input data as a 1-dimensional or 2-dimensional matrix.
			For 2-dimensional matrices, input frames are expected in rows.
			Each row must have a length equal to self.length (as specified
			in the filterbank constructor).
		@return The transformed input data; again in rows, same dtype as input.
		"""
		if len(data.shape) not in (1,2):
			raise ValueError("Only handles 1- and 2-dimensional data, got %d dimensions." % len(data.shape))
		if data.shape[-1] != self.length:
			raise ValueError("Expected data.shape[-1] of %d, got %d." % (self.length, data.shape[-1]))
		# For performance reasons, we handle 1 and 2 dimensions separately
		# (this allows us to use np.dot in the 1D case)
		if len(data.shape) == 1:
			outdata = np.empty(self.num_filters, dtype=data.dtype)
			for b, (l, filt) in enumerate(self._filters):
				outdata[b] = np.dot(data[l:l+len(filt)], filt)
		elif len(data.shape) == 2:
			outdata = np.empty((data.shape[0], self.num_filters), dtype=data.dtype)
			for b, (l, filt) in enumerate(self._filters):
				outdata[:,b] = (data[:,l:l+len(filt)] * filt).sum(axis=1)
		return outdata

