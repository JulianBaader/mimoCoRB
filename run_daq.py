#! /usr/bin/env python3
"""main program controlling DAQ with buffer manager mimoCoRB

   Script to start live data capturing and processing 
   as specified in the 'setup.yaml' file.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import time
import numpy as np
import yaml
from multiprocessing import Process
from mimocorb import  mimo_buffer as bm, appl_lib as ua

class buffer_control():
  """Set-up and management ringbuffers and associated sub-processes
  """

  def __init__(self, buffers_dict, functions_dict, output_directory):
      """
      Class to hold and control mimoCoRB buffer objects 

      :param buffers_dict: dictionary defining buffers RB_1, RB_2, ...
      :param functions_dict: dictionary defining functions FKT_1, FKT_2, ...
      :param output_directory: directory prefix for copies of config files and daq output
      """

      self.buffers_dict = buffers_dict
      self.number_of_ringbuffers = len(buffers_dict) + 1
      self.out_dir = output_directory
      
      self.functions_dict = functions_dict
      self.number_of_functions = len(self.functions_dict)

      self.workers_setup = False
      self.workers_started = False
      
  def setup_buffers(self):
    self.ringbuffers = {}
    for i in range(1, self.number_of_ringbuffers):
        # > Concescutive and concise ring buffer names are assumed! (aka: "RB_1", "RB_2", ...)
        ringbuffer_name = "RB_" + str(i)
        # > Check if the ring buffer exists in the setup_yaml file
        try:
            RB_exists = self.buffers_dict[i-1][ringbuffer_name]
        except KeyError:
            raise RuntimeError("Ring buffer '{}' not found in setup congiguration!\n".format(
                ringbuffer_name))
        num_slots = self.buffers_dict[i-1][ringbuffer_name]['number_of_slots']
        num_ch = self.buffers_dict[i-1][ringbuffer_name]['channel_per_slot']
        data_type = self.buffers_dict[i-1][ringbuffer_name]['data_type']  # simple string type or list expected
        # > Create the buffer data structure (data type of the underlying buffer array)
        if type(data_type) == str:
            rb_datatype = np.dtype(data_type)
        elif type(data_type) == dict:
            rb_datatype = list()
            for key, value in data_type.items():
                rb_datatype.append( (value[0], np.dtype(value[1])) )
        else:
            raise RuntimeError("Ring buffer data type '{}' is unknown! " + 
               "Please use canonical numpy data type names ('float', 'int', 'uint8', ...)" +
               " or a list of tuples (see numpy.dtype()-API reference)".format(data_type))
        
        # > Create and store the new ring buffer object (one per ringbuffer definition in the setup yaml file)
        self.ringbuffers[ringbuffer_name] = bm.NewBuffer(num_slots, num_ch, rb_datatype)
    return self.ringbuffers
    
  def setup_workers(self):
    """Set up all the (parallel) worker functions
    """

    if self.workers_setup:
      print("Cannot setup wokers twice")
      return

    self.process_list = list()

    # get configuration file and runtime
    self.runtime = 0 if 'runtime' not in  self.functions_dict[0]['Fkt_main'] else \
        self.functions_dict[0]['Fkt_main']['runtime']

    if 'config_file' in parallel_functions_dict[0]["Fkt_main"]: 
        cfg_common = parallel_functions_dict[0]['Fkt_main']['config_file']
        # > If common config file is defined: copy it into the target directory ...
        shutil.copyfile(os.path.abspath(cfg_common),
                        os.path.dirname(self.out_dir) + "/" + os.path.basename(cfg_common))
        #    and and load the configuration
        config_dict_common = get_config(cfg_common)
        # if runtime defined, override previous value
        if 'runtime' in config_dict_common['general']: 
            self.runtime = config_dict_common['general']['runtime'] 

    for i in range(1, self.number_of_functions):
        # > Concescutive and concise function names are assumed! (aka: "Fkt_1", "Fkt_2", ...)
        function_name = "Fkt_" + str(i)
        file_py_name = self.functions_dict[i][function_name]['file_name']
        fkt_py_name = self.functions_dict[i][function_name]['fkt_name']
        # print("Load: "+function_name)
        # print(" > "+file_py_name+" "+fkt_py_name)
        number_of_processes = self.functions_dict[i][function_name]['num_process']
        try:
            assigned_ringbuffers = dict(self.functions_dict[i][function_name]['RB_assign'])
        except KeyError:
            assigned_ringbuffers = {}  # no ringbuffer assignment
            # TODO: Do we really want this behaviour? Functions without ring buffers will never
            # receive a 'shutdown()'-signal from the main thread, so they might run indefinitely
            # and block closing the main application
            # (for p in process_list: p.join() blocks until all processes terminate by themselfes!)

        # > Check if this function needs external configuration (specified in a yaml file)
        config_dict = {}
        try:
            # > Use a function specific configuration file referenced in the setup_yaml?
            cfg_file_name = self.functions_dict[i][function_name]['config_file']
        except KeyError:
            # > If there is no specific configuration file, see if there is function specific data
            #   in the common configuration file
            try:
                config_dict = config_dict_common[fkt_py_name]
            except (KeyError, TypeError):
                print("Warning: no configuration found for file '{}'!".format(fkt_py_name))
                pass  # If both are not present, no external configuration is passed to the function 
        else:
            # > In case of a function specific configuration file, copy it over into the target directory
            shutil.copyfile(os.path.abspath(cfg_file_name),
                            os.path.dirname(self.out_dir) + "/" + os.path.basename(cfg_file_name))
            config_dict = get_config(cfg_file_name)

        # > Pass the target-directory created above to the worker function (so, if applicable,
        #   it can safe own data in this directory and everything is contained there)
        config_dict["directory_prefix"] = self.out_dir
        
        # > Prepare function arguments
        source_list = []
        sink_list = []
        observe_list = []

        # > Split ring buffers by usage (as sinks, sources, or observers) and instantiate the
        #   appropriate object to be used by the worker function  (these calls will return
        #   configuration dictionaries used by the bm.Reader(), bm.Writer() or bm.Observer() constructor)
        for key, value in assigned_ringbuffers.items():
            if value == 'read':
                # append new reader dict to the list
                source_list.append(self.ringbuffers[key].new_reader_group())
            elif value == 'write':
                # append new writer dict to the list
                sink_list.append(self.ringbuffers[key].new_writer())
            elif value == 'observe':
                # append new observer dict to the list
                observe_list.append(self.ringbuffers[key].new_observer())

        if not source_list:
            source_list = None
        if not sink_list:
            sink_list = None
        if not observe_list:
            observe_list = None
        # > Create worker processes executing the specified functions in parallel
        parallel_function = self.import_function(file_py_name, fkt_py_name)
        for k in range(number_of_processes):
            self.process_list.append(Process(target=parallel_function,
                                        args=(source_list, sink_list, observe_list, config_dict),
                                        kwargs=assigned_ringbuffers, name=fkt_py_name))

    self.workers_setup = True     

  def start_workers(self):
    """start all of the (parallel) worker functions
    """

    if self.workers_started:
      print("Workers already started")
    
    # > To avoid potential blocking during startup, processes will be started in reverse
    #   data flow order (so last item in the processing chain is started first)
    self.process_list.reverse()
    
    for p in self.process_list:
        p.start()

    self.workers_started = True
    return self.process_list

  @staticmethod
  def import_function(module_path, function_name):
    """
    Import a named object defined in a config yaml file from a module.

    Parameters:
        module_path (str): name of the python module containing the function/class
        function_name (str): python function/class name
    Returns:
        (obj): function/method name callable as object
    Raises:
        ImportError: returns None
    """
    try:
        path = Path(module_path)
        py_module = path.name
        res = path.resolve()
        path_sys = str(res).removesuffix(py_module)  # path to directory
        if path_sys not in sys.path:
            sys.path.append(path_sys)
        module = __import__(py_module, globals(), locals(), fromlist=[function_name])
    except ImportError as ie:
        print("Import Error!", ie)
        return None
    return vars(module)[function_name]

  def display_layout(self):
      print("List of buffers")
      for name, buffer in self.ringbuffers.items():
          print(name,buffer.number_of_slots, buffer.values_per_slot)        
          
  def shutdown(self):
      """Delete buffers, stop processes by calling the shutdown()-Method of the buffer manager
      """
      for name, buffer in self.ringbuffers.items():
        print("Shutting down buffer ",name)
        buffer.shutdown()
        del buffer

      # > All worker processes should have terminated by now
      for p in self.process_list:
          p.join()        

      # > delete remaining ring buffer references (so each buffer managers destructor gets called)
      del self.ringbuffers

  def pause(self):
      """Pause data acquisition
      """
      # disable writing to Buffer RB_1
      self.ringbuffers['RB_1'].pause()

  def resume(self):
      """re-enable  data acquisition
      """
      # disable writing to Buffer RB_1
      self.ringbuffers['RB_1'].resume()
      
# <-- end class buffer_control
    
#helper functions
def get_config(config_file):
    """
    Args:
        config_file: defined in main_setup file (yaml) with fixed name key config_file

    Returns: yaml configuration file content (dict)
    """
    with open(os.path.abspath(config_file), "r") as f:
        config_str = yaml.load(f, Loader=yaml.FullLoader)  # SafeLoader
    return config_str


if __name__ == '__main__': # ---------------------------------------------------------------------------
    
    print("Script: " + os.path.basename(sys.argv[0]))
    ## print("Python: ", sys.version, "\n".ljust(22, '-'))
    
    #  Setup command line arguments and help messages
    parser = argparse.ArgumentParser(
        description="start live data capturing and processing as specified in the 'setup.yaml' file")
    parser.add_argument("setup", type=str)
    arguments = parser.parse_args()

    #  Load setup yaml file
    if not arguments.setup:
        raise RuntimeError("No setup YAML file was specified!")
    else:
        setup_filename = arguments.setup
    try:
        with open(os.path.abspath(setup_filename), "r") as file:
            setup_yaml = yaml.load(file, Loader=yaml.FullLoader)  # SafeLoader
    except FileNotFoundError:
        raise FileNotFoundError("The setup YAML file '{}' does not exist!".format(setup_filename))
    except yaml.YAMLError:
        raise RuntimeError("An error occurred while parsing the setup YAML file '{}'!".format(setup_filename))

    # > Get start time
    start_time = time.localtime()
    
    # > Create the 'target' directory (with setup name and time code)
    template_name = Path(setup_filename).stem
    template_name = template_name[:template_name.find("setup")]
    directory_prefix = "target/" + template_name + "{:04d}-{:02d}-{:02d}_{:02d}{:02d}{:02d}/".format(
        start_time.tm_year, start_time.tm_mon, start_time.tm_mday,
        start_time.tm_hour, start_time.tm_min, start_time.tm_sec )
    os.makedirs(directory_prefix, mode=0o0770, exist_ok=True)
    # > Copy the setup.yaml into the target directory
    shutil.copyfile(os.path.abspath(setup_filename),
                    os.path.dirname(directory_prefix) + "/" + setup_filename)

    # > Separate setup_yaml into ring buffers and functions:
    ringbuffers_dict = setup_yaml['RingBuffer']
    parallel_functions_dict = setup_yaml['Functions']

    # > Hook: possibility to execute user specific code "before ring buffer creation" 
    ua.appl_init()

    # > Set up all needed ring buffers
    bc = buffer_control(ringbuffers_dict, parallel_functions_dict, directory_prefix)
    ringbuffers = bc.setup_buffers()
    print("{:d} buffers created...".format(len(ringbuffers)))
            
    # set-up  workers ...     
    bc.setup_workers()

    bc.display_layout()
    # ... and start all workers     
    process_list = bc.start_workers()
    print("{:d} workers started...".format(len(process_list)))


    # begin of data acquisition -----
    # get runtime defined in config dictionary
    runtime = bc.runtime
    Nprocessed = 0

    now = time.time()
    if runtime != 0:
        # > As sanity check: Print the expected runtime (start date and finish date) in human readable form
        print("Start: ", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(now)),
              " - end: ", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(now + runtime)), '\n')
        while time.time() - now < runtime:
                             # This must not get too small!
                             #  All the buffer managers (multiple threads per ring buffer)
                             #  run in the main thread and may block data flow if execution time constrained!
            time.sleep(0.5)  
            buffer_status = ""
            for RB_name, buffer in ringbuffers.items():
                Nevents, n_filled, rate = buffer.buffer_status()
                if RB_name == 'RB_1': Nprocessed = Nevents
                buffer_status += ': '+ RB_name + " {:3d} ({:d}) {:.4g}Hz) ".format(n_filled, Nevents, rate)
            print("Time remaining: {:.0f}s".format(now + runtime - time.time()) +
                "  - buffers:" + buffer_status, end="\r")
        # when done, first stop data taking          
        bc.pause()
        print("\n      Execution time: {:.2f}s -  Events processed: {:d}".format(
                       int(100*(time.time()-now))/100., Nprocessed) )

    else:  # > 'Batch mode' - processing end defined by an event
           #   (worker process exiting, e.g. no more events from file_source)
        run = True
        print("Batch mode - buffer manager keeps running until one worker process exits!\n")
        animation = ['|', '/', '-', '\\']
        animation_step = 0
        while run:
            time.sleep(0.5)
            buffer_status = ""
            for RB_name, buffer in ringbuffers.items():
                Nevents, n_filled, rate = buffer.buffer_status()
                if RB_name == 'RB_1': Nprocessed = Nevents
                buffer_status += ': '+ RB_name + " {:d} ({:d}) {:.3g}Hz".format(Nevents, n_filled, rate)
            print(" > {} ".format(animation[animation_step]) + buffer_status + 9*' ', end="\r")
            animation_step = (animation_step + 1)%4
            for p in process_list:
                if p.exitcode == 0:  # Wait until one processes exits 
                    run = False
                    break
        # stop data taking          
        bc.pause()    
        print("\n      Execution time: {:.2f}s -  Events processed: {:d}".format(
                       int(100*(time.time()-now))/100., Nprocessed) )
        input("\n\n      Finished - type enter to exit -> ")

    # -->   end of main loop < --

    print("\nSession ended, sending shutdown signal...")

    # -->   finally, shut-down all processes

    # > user defined application run after timer finished (can be ignored if not needed)
    ua.appl_after_start()

    # some grace time for things to finish cleanly ... 
    time.sleep(0.5)
    # ... before shutting down
    bc.shutdown()

    # > user defined application run after all processing is finished (can be ignored if not needed)
    ua.appl_after_stop()

    print("      Finished - Good Bye")
