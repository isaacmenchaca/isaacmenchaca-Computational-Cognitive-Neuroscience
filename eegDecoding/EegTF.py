import numpy as np
from numpy import log10
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
from mpl_toolkits import mplot3d

__ALL__ = ['stft4EegChannel', 'cwtMorl4EEGChannel']

def _stft_density(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None,
         detrend=False, return_onesided=True, boundary='zeros', padded=True,
         axis=-1):
  

    freqs, time, Zxx = signal.spectral._spectral_helper(x, x, fs, window, nperseg, noverlap,
                                        nfft, detrend, return_onesided,
                                        scaling='density', axis=axis,
                                        mode='stft', boundary=boundary,
                                        padded=padded)

    return freqs, time, Zxx


def stft4EegChannel(fileName: str, channel: str, timeStep: int, 
                   startTime: int, endTime: int, windowSize: int, 
                    desiredStartTime: int = None, desiredEndTime: int = None,
                    trial_epoch: int = None,
                   frequencyStop: int = None, 
                    windowTaper:str = 'hann', log_power: bool = True, 
                    plot_data: str = 'All', 
                    scaling='spectrum'):
    
    data = scipy.io.loadmat(fileName)
    EEGdata = data['EEG']['data'][0][0] # data points
    EEGtimes = data['EEG']['times'][0][0][0] # EEG sample times
    EEGsrate = float(data['EEG']['srate'][0][0][0][0]) # sampling rate
    
    EEGtrials = data["EEG"][0][0]["trials"][0][0] # epochs
    channelLocLabels = data['EEG']['chanlocs'][0][0][0]['labels'] # possible channels
    EEGnbchan = data["EEG"][0,0]["nbchan"][0,0] # number of channels
    
    channel = 'Fz' # specified label of channel, its location will set to true.
    channelIndex = channelLocLabels == channel # selected channel from list of channels
    
    
    if type(trial_epoch) == int and trial_epoch <= EEGtrials and np.ndim(EEGdata[channelIndex, :]) == 3:
        detrended_Pnts = signal.detrend(np.squeeze(EEGdata[channelIndex, :, 
                                                           trial_epoch - 1]))
    elif trial_epoch == None:
        if np.ndim(EEGdata[channelIndex, :]) == 3:
            detrended_Pnts = signal.detrend(np.squeeze(EEGdata[channelIndex, :, 0]))
        else:
            detrended_Pnts = signal.detrend(np.squeeze(EEGdata[channelIndex, :]))
            
    
    
    if desiredStartTime == None:
        startEndTime = np.array([startTime, endTime])
    else:
        startEndTime = np.array([desiredStartTime, desiredEndTime])
    
    saveTime = np.zeros(startEndTime.shape)
    for i in range(len(startEndTime)):
        saveTime[i] = np.argmin(np.absolute(EEGtimes-startEndTime[i]))
        
    if scaling == 'spectrum':
        f, t, Zxx = signal.stft(detrended_Pnts[int(saveTime[0]):int(saveTime[1])+1], 
                        EEGsrate, nperseg=windowSize, window = windowTaper,
                           noverlap= windowSize - timeStep)
        
    elif scaling == 'density':
        f, t, Zxx = _stft_density(detrended_Pnts[int(saveTime[0]):int(saveTime[1])+1], 
                        EEGsrate, nperseg=windowSize, window = windowTaper,
                       noverlap= windowSize - timeStep)
    else:
         raise ValueError('unknown value for mode')
    
    
    tf_time = np.linspace(startTime, endTime, t.shape[0])
    
    if frequencyStop == None:
        # move on
        tf_frequency = f.copy()
        if log_power:
            tf_log_power = log10(np.abs(Zxx.copy()))
        else: 
            tf_log_power = (np.abs(Zxx.copy()))


    elif frequencyStop < EEGsrate / 2: # less than nyquist.
        # filter the frequency.
        stftFilter = np.argmin(np.absolute(f - frequencyStop))
        tf_frequency = f[:stftFilter+1].copy()
        if log_power:
            tf_log_power = log10(np.abs(Zxx[:stftFilter+1]))
        else:
            tf_log_power = (np.abs(Zxx[:stftFilter+1]))

    
    if plot_data != None:
        fig = plt.figure(figsize=(15, 15))
    
    sns.set_palette('muted')
    sns.set_style('darkgrid')
    
    if plot_data == 'All_cb' or  plot_data == 'All': # plot eeg data and stft with ow w/o colorbar.
        plt.subplot(2, 1, 1)
        plt.plot(EEGtimes[int(saveTime[0]):int(saveTime[1])+1],
         detrended_Pnts[int(saveTime[0]):int(saveTime[1])+1])
        
        ymin = np.amin(detrended_Pnts[int(saveTime[0]):int(saveTime[1])+1]) - 10
        ymax = np.amax(detrended_Pnts[int(saveTime[0]):int(saveTime[1])+1]) + 10
        xmin = np.amin(EEGtimes[int(saveTime[0]):int(saveTime[1])+1])
        xmax = np.amax(EEGtimes[int(saveTime[0]):int(saveTime[1])+1])


        plt.vlines(x = 0,ymin = ymin, ymax = ymax, linestyles='dashed')
        plt.hlines(y = 0,xmin = xmin, xmax = xmax, linestyles='dashed')

        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])
        plt.ylabel(r'$ \mu V $')
        plt.xlabel('Time (ms)')
        
        
        if trial_epoch != None:
            plt.title('EEG signal of channel %s - Trial: %i' % (channel, trial_epoch))
        else:
            plt.title('EEG signal of channel %s' % channel)
        
        plt.subplot(2, 1, 2)
        plt.pcolormesh(tf_time, tf_frequency, 
               tf_log_power,cmap=plt.cm.jet)

        ymax = np.amax(tf_frequency)
        plt.vlines(x = 0,ymin = 0, ymax = ymax, linestyles='dashed')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (Hz)')
        
        plt.title('STFT TF Decomposition')
        if plot_data == 'All_cb': # plot eeg data and stft without colorbar.
            clb = plt.colorbar()
            if scaling == 'spectrum':
                if log_power:
                    clb.ax.set_title('log10(uV ** 2)')
                else: 
                    clb.ax.set_title('uV ** 2')
            elif scaling == 'density':
                if log_power:
                    clb.ax.set_title('log10(uV ** 2 / Hz)')
                else: 
                    clb.ax.set_title('uV ** 2 / Hz')
        

    elif plot_data == 'tf': # plot tf only.
        plt.pcolormesh(tf_time, tf_frequency, 
               tf_log_power,cmap=plt.cm.jet)

        ymax = np.amax(tf_frequency)
        plt.vlines(x = 0,ymin = 0, ymax = ymax, linestyles='dashed')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (Hz)')
        plt.title('STFT TF Decomposition')
        clb = plt.colorbar()
        
        if scaling == 'spectrum':
            if log_power:
                clb.ax.set_title('log10(uV ** 2)')
            else: 
                clb.ax.set_title('uV ** 2')
        elif scaling == 'density':
            if log_power:
                clb.ax.set_title('log10(uV ** 2 / Hz)')
            else: 
                clb.ax.set_title('uV ** 2 / Hz')
        
    elif plot_data == 'tf_3d': # plot tf only in 3d.
        sns.set_style('whitegrid')
        x, y = np.meshgrid(tf_time, tf_frequency) # make a meshgrid out of x and y
        z = tf_log_power.copy()
        
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap=plt.cm.jet)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (Hz)')
        plt.suptitle('3D STFT TF Decomposition')
        
        if scaling == 'spectrum':
            if log_power:
                ax.set_zlabel('power (log10(uV ** 2))')
            else: 
                ax.set_zlabel('power (uV ** 2)')  
        elif scaling == 'density':
            if log_power:
                ax.set_zlabel('power (log10(uV ** 2 / Hz))')
            else:
                ax.set_zlabel('power (uV ** 2 / Hz)')



    if plot_data != None:
        if desiredStartTime != None:
            plt.xlim([desiredStartTime, desiredEndTime])
        else:  
            plt.xlim([startTime, endTime])
        
        plt.show()

    return tf_frequency, tf_time, tf_log_power





def cwtMorl4EEGChannel(fileName: str, channel: str, startTime: int, endTime: int, trial_epoch: int = None, frequencyStop: int = None, log_power: bool = True, w: float = 4., plot_data: str = 'All'):
        
        data = scipy.io.loadmat(fileName)
        EEGdata = data['EEG']['data'][0][0] # data points
        EEGtimes = data['EEG']['times'][0][0][0] # EEG sample times
        EEGsrate = float(data['EEG']['srate'][0][0][0][0]) # sampling rate
    
        EEGtrials = data["EEG"][0][0]["trials"][0][0] # epochs
        channelLocLabels = data['EEG']['chanlocs'][0][0][0]['labels'] # possible channels
        EEGnbchan = data["EEG"][0,0]["nbchan"][0,0] # number of channels
    
        channel = 'Fz' # specified label of channel, its location will set to true.
        channelIndex = channelLocLabels == channel # selected channel from list of channels
    
        print('EEG data shape (channels, data points, epochs):', EEGdata.shape)
        print('Time:', EEGtimes[0], 'to', EEGtimes[-1])
        print('Sampling rate:', EEGsrate)
        print('Num. of Trial Epochs:', EEGtrials)
        print('Num. of Channels:', EEGnbchan)
        print('Channel selected:', channel)
    
    
        if type(trial_epoch) == int and trial_epoch <= EEGtrials and np.ndim(EEGdata[channelIndex, :]) == 3:
            detrended_Pnts = signal.detrend(np.squeeze(EEGdata[channelIndex, :, 
                                                           trial_epoch - 1]))
        elif trial_epoch == None:
            if np.ndim(EEGdata[channelIndex, :]) == 3:
                detrended_Pnts = signal.detrend(np.squeeze(EEGdata[channelIndex, :, 0]))
            else:
                detrended_Pnts = signal.detrend(np.squeeze(EEGdata[channelIndex, :]))
        startEndTime = np.array([startTime, endTime]) 
        
        saveTime = np.zeros(startEndTime.shape)
        for i in range(len(startEndTime)):
            saveTime[i] = np.argmin(np.absolute(EEGtimes-startEndTime[i]))
            
        if frequencyStop == None: 
            tf_frequency = np.linspace(1, EEGsrate / 2, EEGsrate / 2)
        elif frequencyStop < EEGsrate / 2:
            tf_frequency = np.linspace(1, frequencyStop, frequencyStop)
            
        widths = w*EEGsrate / (2*tf_frequency*np.pi)
        
        cwtm = signal.cwt(detrended_Pnts[int(saveTime[0]):int(saveTime[1])+1], 
                  signal.morlet2, widths, w = w)
        
         
        tf_time = EEGtimes[int(saveTime[0]):int(saveTime[1])+1]
        
        if log_power:
            tf_power = log10(np.abs(cwtm))
        else:
            tf_power = (np.abs(cwtm))
            
        if plot_data != None:
            fig = plt.figure(figsize=(15, 15))
    
        sns.set_palette('muted')
        sns.set_style('darkgrid')
        
        
        if plot_data == 'All_cb' or  plot_data == 'All': 
            plt.subplot(2, 1, 1)
            plt.plot(EEGtimes[int(saveTime[0]):int(saveTime[1])+1],
             detrended_Pnts[int(saveTime[0]):int(saveTime[1])+1])
        
            ymin = np.amin(detrended_Pnts[int(saveTime[0]):int(saveTime[1])+1]) - 10
            ymax = np.amax(detrended_Pnts[int(saveTime[0]):int(saveTime[1])+1]) + 10
            xmin = np.amin(EEGtimes[int(saveTime[0]):int(saveTime[1])+1])
            xmax = np.amax(EEGtimes[int(saveTime[0]):int(saveTime[1])+1])


            plt.vlines(x = 0,ymin = ymin, ymax = ymax, linestyles='dashed')
            plt.hlines(y = 0,xmin = xmin, xmax = xmax, linestyles='dashed')

            plt.ylim([ymin, ymax])
            plt.xlim([xmin, xmax])
            plt.ylabel(r'$ \mu V $')
            plt.xlabel('Time (ms)')
        
        
            if trial_epoch != None:
                plt.title('EEG signal of channel %s - Trial: %i' % (channel, trial_epoch))
            else:
                plt.title('EEG signal of channel %s' % channel)
        
            plt.subplot(2, 1, 2)
            plt.pcolormesh(tf_time, tf_frequency, 
                   tf_power, cmap=plt.cm.jet)
        

            ymax = np.amax(tf_frequency)
            plt.vlines(x = 0,ymin = 1, ymax = ymax, linestyles='dashed')
            plt.xlabel('Time (ms)')
            plt.ylabel('Frequency (Hz)')
        
            plt.title('Morlet Wavelet TF Decomposition')
            
            if plot_data == 'All_cb': # plot eeg data and stft without colorbar.
                clb = plt.colorbar()
                if log_power:
                    clb.ax.set_title('log10(uV ** 2 / Hz)')
                else: 
                    clb.ax.set_title('uV ** 2 / Hz')
                
         
        elif plot_data == 'tf': # plot tf only.
            plt.pcolormesh(tf_time, tf_frequency, 
                   tf_power, cmap=plt.cm.jet)

            ymax = np.amax(tf_frequency)
            plt.vlines(x = 0,ymin = 1, ymax = ymax, linestyles='dashed')
            plt.xlabel('Time (ms)')
            plt.ylabel('Frequency (Hz)')
        
            plt.title('Morlet Wavelet TF Decomposition')
            
            clb = plt.colorbar()
        
            if log_power:
                clb.ax.set_title('log10(uV ** 2 / Hz)')
            else: 
                clb.ax.set_title('uV ** 2 / Hz')
                        
        elif plot_data == 'tf_3d': # plot tf only in 3d.
            sns.set_style('whitegrid')
            x, y = np.meshgrid(tf_time, tf_frequency) # make a meshgrid out of x and y
            z = tf_power.copy()
        
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(x, y, z, cmap=plt.cm.jet)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Frequency (Hz)')
            plt.suptitle('3D Morlet Wavelet TF Decomposition')
        
            if log_power:
                ax.set_zlabel('power (log10(uV ** 2 / Hz))')
            else:
                ax.set_zlabel('power (uV ** 2 / Hz)')
                
        plt.xlim([startTime, endTime])
        
        plt.show()
      
        
        return tf_frequency, tf_time, tf_power
        
        
