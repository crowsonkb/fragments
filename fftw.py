import os
import pyfftw

class BasicFFT:
    def __init__(self, size):
        self.size = size
        self._time = pyfftw.empty_aligned(size, 'float64')
        self._freq = pyfftw.empty_aligned(size//2 + 1, 'complex128')
        self.fft = pyfftw.FFTW(self._time, self._freq, threads=os.cpu_count(),
                               direction='FFTW_FORWARD')
        self.ifft = pyfftw.FFTW(self._freq, self._time, threads=os.cpu_count(),
                                direction='FFTW_BACKWARD')
