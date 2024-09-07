"""
rb_to_spectrum is used to histogram data from a buffer and save it to file(s).
plot can be used to plot the histogram data in real-time.
If plot is provided with an observer, the observer data will be plotted in the same figure as an oscilloscope.

Intended usecase would be
RB_i contains raw oscilloscope data
RB_i+1 contains an analyzed metric like, pulse height, pulse width, etc.
The RB_i+1 buffer must have a large enough number_of_slots as reading will halt regularly to save the file.
-> TODO copy data to a buffer and handle saving in a different thread?

In this case, it might be interesting to add a visualization of the analyzed metric to the oscilloscope data.
But this has to be added on a case-by-case basis. -> TODO provide easy interface for this

It is recommended to use the same config files for both functions containing:

update_interval -> time between updates of the histogram
filename -> name of the file to save the histogram data (will be appended with the key)
spectrum:
    key: min, max, nbins -> key and range of the histogram for each key to be histogrammed

if oscilloscope is used:
update_interval_osc -> time between updates of the oscilloscope
osc_range -> range of the oscilloscope will be -osc_range to osc_range
"""

from .buffer_control import rbObserver
from .buffer_control import rbExport
import numpy as np
import matplotlib.pyplot as plt
import time
import os


def rb_to_spectrum(source_list=None, sink_list=None, observe_list=None, config_dict=None, **rb_info):
    # get information from source_list
    if len(source_list) > 1:
        print("!!! More than one source presently not foreseen!!")
    available_keys = [dtype[0] for dtype in source_list[0]['dtype']]
    exporter = rbExport(source_list=source_list, config_dict=config_dict, **rb_info)

    # get information from config_dict

    update_interval = config_dict['update_interval']
    filename = config_dict['directroy_prefix'] + config_dict['filename']
    keys = config_dict['spectrum'].keys()

    for key in keys:
        if key not in available_keys:
            print(f"!!! Key {key} not available in source data and will be ignored!!")
            keys.remove(key)

    min_vals = {}
    max_vals = {}
    nbins = {}
    factors = {}
    hists = {}
    discardeds = {}
    for key in keys:
        min_vals[key] = config_dict['spectrum'][key][0]
        max_vals[key] = config_dict['spectrum'][key][1]
        nbins[key] = config_dict['spectrum'][key][2]
        factors[key] = (max_vals[key] - min_vals[key]) / nbins[key]
        hists[key] = np.zeros(nbins[key])
        discardeds[key] = []

    last_update_time = time.time()

    while True:
        d = next(exporter(), None)  # this blocks until new data provided !
        if d is None:  # last data received -> finish execution
            save_spectrum(hists, filename)
            save_spectrum(discardeds, filename + '_discarded_')
            break
        data, metadata = d
        for key in keys:
            vals = data[key]
            for val in vals:
                if val is None:
                    continue
                index = int((val - min_vals[key]) * factors[key])
                if 0 <= index < nbins[key]:
                    hists[key][int((val - min_vals[key]) * factors[key])] += 1
                else:
                    discardeds[key].append(val)

        if time.time() - last_update_time > update_interval:
            last_update_time = time.time()
            save_spectrum(hists, filename)


def save_spectrum(hists, filename):
    for key, hist in hists.items():
        np.save(filename + key, hist)


def plot(source_list=None, sink_list=None, observe_list=None, config_dict=None, **rb_info):
    # get information from observe_list
    show_osc = observe_list is not None
    if len(observe_list) > 1:
        print("!!! More than one observe presently not foreseen!!")
    if show_osc:
        keys_osc = [dtype[0] for dtype in observe_list[0]['dtype']]
        channels = observe_list[0]['values_per_slot']
        osc_observer = rbObserver(observe_list=observe_list, config_dict=config_dict, **rb_info)

    # get information from config_dict
    update_interval = config_dict['update_interval']
    filename = config_dict['directroy_prefix'] + config_dict['filename']
    keys = config_dict['spectrum'].keys()
    spectrum_config = config_dict['spectrum']

    if show_osc:
        update_interval_osc = config_dict['update_interval_osc']
        osc_range = config_dict['osc_range']

    fig = plt.figure()

    # initialize oscilloscope
    if show_osc:
        ax_osci = fig.add_subplot(211)
        ax_osci.set_title('Oscilloscope')
        x = np.arange(0, channels)
        y = np.zeros(channels)
        lines_osc = {}
        for key in keys_osc:
            (lines_osc[key],) = ax_osci.plot(x, y, label=key)
        ax_osci.set_xlim(0, channels)
        ax_osci.set_ylim(-osc_range, osc_range)

    # initialize spectrum
    if show_osc:
        ax_spectrum = fig.add_subplot(212)
    else:
        ax_spectrum = fig.add_subplot(111)

    ax_spectrum.set_title('Spectrum')
    xs = {}
    ys = {}
    lines = {}
    for key in keys:
        xs[key] = np.arange(spectrum_config[key]['min'], spectrum_config[key]['max'], spectrum_config[key]['nbins'])
        ys[key] = np.zeros(spectrum_config[key]['nbins'])
        lines[key] = ax_spectrum.plot(xs[key], ys[key], label=key)

    plt.ion()
    plt.show()

    last_update_time = time.time()
    last_update_time_osc = time.time()

    while True:
        if time.time() - last_update_time > update_interval:
            # update spectrum
            for key in keys:
                if os.path_exists(filename + key + '.npy'):
                    ys[key] = np.load(filename + key + '.npy')
                    lines[key][0].set_ydata(ys[key])
            ax_spectrum.relim()
            ax_spectrum.autoscale_view()
            fig.canvas.draw()
        if show_osc and time.time() - last_update_time_osc > update_interval_osc:
            # update oscilloscope
            data = next(osc_observer(), None)
            if data is None:
                break
            for key, line in lines_osc.items():
                line.set_ydata(data[0][key])
            fig.canvas.draw()

        fig.canvas.flush_events()
        time.sleep(0.03)  # to avoid too much CPU usage but still provide 30FPS
