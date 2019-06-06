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

from mpi4py import MPI

from npy2mpi import fromnumpy, tonumpy

class AudioStream(object):
    def __init__(self):

        #INITILIZE BASIC MPI DATA
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        print("SIZE: ",self.size)
        print("RANK: ",self.rank)

        # pyaudio stuff CHUNK
        self.CHUNK = 1024 * 2

        #INITIALIZE MPI DATA

        self.sizeOfChunk = self.CHUNK // self.size
        #if number of processors isn't CHUNK%num ==0
        #then last process takes remainding array elements
        self.sizeOfLastChunk = self.CHUNK - self.sizeOfChunk * (self.size - 1)
        self.senddata = None



        # recvdata is array of split data after Scatterv
        if self.rank != self.size - 1:
            self.recvdata = np.zeros(self.sizeOfChunk, dtype=np.int16)
        else:
            self.recvdata = np.zeros(self.sizeOfLastChunk, dtype=np.int16)

        self.counts = ()
        self.dspls = ()
        self.countsMinusOne = (self.sizeOfChunk,) * (self.size - 1)
        self.counts = self.countsMinusOne + (self.sizeOfLastChunk,)
        print("Counts: ", self.counts)
        for i in range(0, self.size):
            self.dspls = self.dspls + (i * self.sizeOfChunk,)
        print("Dspls: ", self.dspls)


       # if self.rank == 0:
        self.initializeBasicData()


      # self.comm.Barrier()

        # # senddata is data to be split and then send to processes
        # if self.rank == 0:
        #     self.senddata = np.arange(self.CHUNK, dtype=np.int)

        print("END OF INITIALIZATION")
    def initializeBasicData(self):
        # pyqtgraph stuff
        if self.rank == 0:
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
        #if self.rank ==0:
        wf_data = self.stream.read(self.CHUNK)
       # wf_data = struct.unpack(str(2 * self.CHUNK) + 'B', wf_data)
       #wf_data = np.array(wf_data, dtype='b')[::2] + 128

        wf_data = np.frombuffer(wf_data, dtype=np.int16)/65536*255 * self.waveMultiplier + 128
        print("WF_data_start = ", len(wf_data))
        self.senddata=wf_data

        #dataType=fromnumpy(np.dtype(np.int16))
        dataType=MPI.INT16_T
        try:
            self.comm.Scatterv([wf_data, self.counts, self.dspls, dataType], self.recvdata, root=0)
        except Exception as ex:
            print("Exception: ", ex)
       # print('on task', self.rank, 'after Scatterv:    data = ', self.recvdata)
       # print(self.recvdata)
        #print("Len: ",len(self.recvdata))
        recvD2 = None
        if self.rank == 0:
            recvD2 = np.zeros(self.CHUNK, dtype=np.int16)
       # print("recvD2 len: ", len(recvD2))




        #END MPI

        if self.rank == 0:
            self.set_plotdata(name='waveform', data_x=self.x, data_y=wf_data)


        if self.isFFT:
            sp_data = fft(np.array(self.recvdata, dtype='int8') - 128)
            sp_data = np.abs(sp_data[0:int(self.CHUNK / 2)]) * 2 / (128 * self.CHUNK)
            try:
                self.comm.Gatherv(sp_data, [recvD2, self.counts, self.dspls, MPI.SHORT])
            except Exception as ex:
                print("Exception: ", ex)
            self.set_plotdata(name='spectrum', data_x=self.f, data_y=sp_data)
        else:
            widths = np.arange(1, 31)
            waveletReturn = signal.cwt(np.array(self.recvdata, dtype='int8'), signal.ricker, widths)

#fromnumpy(np.dtype(np.int16))
            try:
                self.comm.Gatherv(self.recvdata, [recvD2, self.counts, self.dspls, dataType])
                print(len(recvD2))
            except Exception as ex:
                print("Exception: ", ex)

            self.set_plotdata(name='spectrum', data_x=self.f, data_y=recvD2[ :1024])





        # print(recvD2)
        # if self.rank == 0:
        # print("On task ", self.rank, "after Gatherv:    data = ", recvD2)

    def animation(self):
       # self.a=0
        timer = None
        #if self.rank == 0:
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(60)

       # if self.rank == 0:
        self.start()


if __name__ == '__main__':


    audio_app = AudioStream()
    audio_app.animation()

