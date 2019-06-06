
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import struct
import pyaudio
from scipy.fftpack import fft
from scipy import signal
import sys
import time

import matplotlib.pyplot as plt

class AudioStream(object):
    def __init__(self):

        # pyqtgraph stuff
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        self.app = QtGui.QApplication(sys.argv)
        self.win = pg.GraphicsWindow(title='Spectrum Analyzer')
        self.win.setWindowTitle('Spectrum Analyzer')
        self.win.setGeometry(5, 115, 1910, 1070)

        wf_xlabels = [(0, '0'), (2048, '2048'), (4096, '4096')]
        wf_xaxis = pg.AxisItem(orientation='bottom')
        wf_xaxis.setTicks([wf_xlabels])

        wf_ylabels = [(0, '0'), (127, '128'), (255, '255')]
        wf_yaxis = pg.AxisItem(orientation='left')
        wf_yaxis.setTicks([wf_ylabels])

        self.isLogAxis = False
        self.waveMultiplier = 4
        self.isFFT = False

        if self.isFFT:
            if self.isLogAxis:
                sp_xlabels = [
                    (np.log10(10), '10'), (np.log10(100), '100'),
                    (np.log10(1000), '1000'), (np.log10(22050), '22050')
                ]
            else:
                sp_xlabels = [
                    (10, '10'), (1000, '1000'),
                    (10000, '10000'), (20000, '20000')
                ]
        else:
            sp_xlabels = [
                (10, '10'), (1000, '1000'),
                (10000, '10000'), (15000, '15000')
            ]

        sp_xaxis = pg.AxisItem(orientation='bottom')
        sp_xaxis.setTicks([sp_xlabels])
        self.waveform = self.win.addPlot(
            title='WAVEFORM', row=1, col=1, axisItems={'bottom': wf_xaxis, 'left': wf_yaxis},
        )
        if self.isFFT:
            self.spectrum = self.win.addPlot(
                title='SPECTRUM', row=2, col=1, axisItems={'bottom': sp_xaxis},
            )
        else:
            self.spectrum = self.win.addPlot(
                title='Continous wavelet function result', row=2, col=1, axisItems={'bottom': sp_xaxis},
            )

        # pyaudio stuff
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )
        # waveform and spectrum x points
        self.x = np.arange(0, 2 * self.CHUNK, 2)
        self.f = np.linspace(0, self.RATE / 2, self.CHUNK / 2)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def set_plotdata(self, name, data_x, data_y):
        if name in self.traces:
            self.traces[name].setData(data_x, data_y)
        else:
            if name == 'waveform':
                self.traces[name] = self.waveform.plot()
                self.traces[name].setPen(color='c', width=1)
                self.waveform.setYRange(0, 255, padding=0)
                self.waveform.setXRange(0, 2 * self.CHUNK, padding=0.005)
            if name == 'spectrum':
                self.traces[name] = self.spectrum.plot()
                self.traces[name].setPen(color='c', width=1)

                if self.isFFT:
                    if self.isLogAxis:
                        self.spectrum.setLogMode(x=True, y=True)
                        self.spectrum.setYRange(-4, 0, padding=0)
                        self.spectrum.setXRange(
                            np.log10(20), np.log10(self.RATE / 2), padding=0.005)
                    else:
                        self.spectrum.setLogMode(x=False, y=False)
                        if self.isFFT:
                            self.spectrum.setYRange(0, 3, padding=0)
                            self.spectrum.setXRange(20, 20000, padding=0.005)
                else:
                    self.spectrum.setLogMode(x=False, y=False)
                    self.spectrum.setYRange(-255, 255, padding=0)
                    self.spectrum.setXRange(20, 10000, padding=0.005)

    def update(self):
        wf_data = self.stream.read(self.CHUNK)
       # wf_data = struct.unpack(str(2 * self.CHUNK) + 'B', wf_data)
       #wf_data = np.array(wf_data, dtype='b')[::2] + 128

        wf_data = np.frombuffer(wf_data, dtype=np.int16)/65536*255 * self.waveMultiplier + 128
        self.set_plotdata(name='waveform', data_x=self.x, data_y=wf_data)

        if self.isFFT:
            sp_data = fft(np.array(wf_data, dtype='int8') - 128)
            sp_data = np.abs(sp_data[0:int(self.CHUNK / 2)]) * 2 / (128 * self.CHUNK)
            self.set_plotdata(name='spectrum', data_x=self.f, data_y=sp_data)
        else:
            widths = np.arange(1, 31)
            waveletReturn = signal.cwt(np.array(wf_data, dtype='int8'), signal.ricker, widths)
            print("wavelet len: ", len(waveletReturn))
           # plt.imshow(waveletReturn, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto', vmax = abs(waveletReturn).max(), vmin = -abs(waveletReturn).max())
            #plt.show()

           # if self.a==30:
           #     self.a=0
            self.set_plotdata(name='spectrum', data_x=self.f, data_y=waveletReturn[4, :1024])
           # self.a=self.a+1

    def animation(self):
       # self.a=0
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(60)
        self.start()


if __name__ == '__main__':

    audio_app = AudioStream()
    audio_app.animation()

