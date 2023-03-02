"""
**histogram_buffer** collection of classes to produce histograms

Show animated histogram display of scalar buffer variables

code adapted from https://github.com/GuenterQuast/picoDAQ
"""

import time, numpy as np
import itertools

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt, matplotlib.animation as anim

class animHists(object):
  ''' display histogram, as normalised frequency distibutions

      frequency distribution of a scalar quantity
  '''

  def __init__(self, Hdescr, name='Histograms', fig=None):
    ''' 
      Args:
        list of histogram descriptors, 
        where each descriptor is a list itself: [min, max, nbins, ymax, name, type]
        min: minimum value
        max: maximum value
        nbins: nubmer of bins
        ymax:  scale factor for bin with highest number of entries
        name: name of the quantity being histogrammed
        type: 0 linear, 1 for logarithmic y scale
        name forfigure window
    '''
  
    self.nHist = len(Hdescr)
    self.entries = np.zeros(self.nHist)
    self.frqs = []
    
  # histrogram properties
    self.mins = []
    self.maxs = []
    self.nbins = []
    self.ymxs = []
    self.names = []
    self.types = []
    self.bedges = []
    self.bcents = []
    self.widths = []
    for ih in range(self.nHist):
      self.mins.append(Hdescr[ih][0])
      self.maxs.append(Hdescr[ih][1])
      self.nbins.append(Hdescr[ih][2])
      self.ymxs.append(Hdescr[ih][3])
      self.names.append(Hdescr[ih][4])
      self.types.append(Hdescr[ih][5])
      be = np.linspace(self.mins[ih], self.maxs[ih], self.nbins[ih] +1 ) # bin edges
      self.bedges.append(be)
      self.bcents.append( 0.5*(be[:-1] + be[1:]) )                       # bin centers
      self.widths.append( 0.5*(be[1]-be[0]) )                           # bar width

  # create figure
    ncols = int(np.sqrt(self.nHist))
    nrows = ncols
    if ncols * nrows < self.nHist: nrows +=1
    if ncols * nrows < self.nHist: ncols +=1

    if fig is None:
      sf = 1. if ncols *nrows != 1 else 2.
      self.fig = plt.figure(name, figsize=(sf*3.*ncols, sf*2.*nrows) )
      axarray = self.fig.subplots(nrows=nrows, ncols=ncols)
      self.fig.subplots_adjust(left=0.25/ncols, bottom=0.25/nrows, right=0.975, top=0.95,
                             wspace=0.35, hspace=0.35)
    else:
        self.fig = fig

  # sort axes in linear array
    self.axes = []
    if self.nHist == 1:
      self.axes = [axarray]
    elif self.nHist == 2:
      for a in axarray:
        self.axes.append(a)
    else:
      nh = 0
      for ir in range(nrows):
        for ic in range(ncols):
          nh += 1
          if nh <= self.nHist:
            self.axes.append(axarray[ir,ic])
          else:
            axarray[ir,ic].axis('off')

    for ih in range(self.nHist):
      self.axes[ih].set_ylabel('frequency')
      self.axes[ih].set_xlabel(self.names[ih])
# guess an appropriate y-range for normalized histogram
      if self.types[ih]:            # log plot
        self.axes[ih].set_yscale('log')
        ymx=self.ymxs[ih]/self.nbins[ih] 
        self.axes[ih].set_ylim(1E-3 * ymx, ymx) 
        self.frqs.append(1E-4*ymx*np.ones(self.nbins[ih]) )
      else:                         # linear y scale
        self.axes[ih].set_ylim(0., self.ymxs[ih]/self.nbins[ih])
        self.frqs.append(np.zeros(self.nbins[ih]))
    
  def init(self):
    self.rects = []
    self.animtxts = []
    for ih in range(self.nHist):
    # plot an empty histogram
      self.rects.append(self.axes[ih].bar( self.bcents[ih], self.frqs[ih], 
           align='center', width=self.widths[ih], facecolor='midnightblue', alpha=0.7) )       
    # emty text
      self.animtxts.append(self.axes[ih].text(0.5, 0.925 , ' ',
              transform=self.axes[ih].transAxes,
              size='small', color='darkred') )

    graf_objects = tuple(self.animtxts) \
              + tuple(itertools.chain.from_iterable(self.rects) )  
    return graf_objects # return tuple of graphics objects

  def __call__(self, vals):

    if vals is None:
      # return old values if no new data
      return tuple(self.animtxts)  \
        + tuple(itertools.chain.from_iterable(self.rects) ) 
      
    # add recent values to frequency array, input is a list of arrays
    for ih in range(self.nHist):
      vs = vals[ih]
      self.entries[ih] += len(vs)
      for v in vs:
        iv = int(self.nbins[ih] * (v-self.mins[ih]) / (self.maxs[ih]-self.mins[ih]))
        if iv >=0 and iv < self.nbins[ih]:
          self.frqs[ih][iv]+=1
      if(len(vs)):
        norm = np.sum(self.frqs[ih]) # normalisation to one
    # set new heights for histogram bars
        for rect, frq in zip(self.rects[ih], self.frqs[ih]):
          rect.set_height(frq/norm)
    # update text
        self.animtxts[ih].set_text('Entries: %i'%(self.entries[ih]) )

    return tuple(self.animtxts)  \
        + tuple(itertools.chain.from_iterable(self.rects) ) 

def plot_Histograms(Q, Hdescripts, interval, name = 'Histograms'):
  ''' show animated histogram(s)
    Args:
      Q:    multiprocessing.Queue() 
      Hdescripts:  list of histogram descriptors, where each 
        descriptor is itself a list: [min, max, nbins, ymax, name, type]
          min: minimum value
          max: maximum value
          nbins: nubmer of bins
          ymax:  scale factor for bin with highest number of entries
          name: name of the quantity being histogrammed
          type: 0 linear, 1 for logarithmic y scale
  '''

  # Generator to provide data to animation
  def yieldData_fromQ():
  # receive data from multiprocessing Queue 
    cnt = 0
    try:
      while True:
#        while not Q.qsize(): 
#          time.sleep(0.1)
        if not Q.qsize():
          yield(None)
        else:  
          v = Q.get(timeout=0.1)
          yield v
        cnt+=1
    except:
      print('*==* yieldData_fromQ: termination signal received')
      return


# ------- executable part -------- 
#  print(' -> plot_Histograms starting')

#  try:
  H = animHists(Hdescripts, name)
  figH = H.fig
# set up matplotlib animation
  Hanim = anim.FuncAnimation(figH, H, yieldData_fromQ, 
                      init_func=H.init, interval=interval, blit=True,
                      fargs=None, repeat=True, save_count=None)
                           # save_count=None is a (temporary) work-around 
                           #     to fix memory leak in animate
  plt.show()
  
#  except:
#    print('*==* plot_Histgrams: termination signal recieved')
  sys.exit()
