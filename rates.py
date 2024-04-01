import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 13})
import re

CLOCK_SPEED = 2.4e-8

class Edge:
  def __init__(self, hex_string, CPLD_time):
    binary_string = format(int(hex_string, 16), "08b")
    self.dt = int(binary_string[3:], 2)
    self.valid = (int(binary_string[2]) == 1)
    self.t = CPLD_time + self.dt * CLOCK_SPEED / 32.0
    # print(binary_string, self.valid)

class Timeline:
  def __init__(self, channelA, channelB):
    self.IDA = str(channelA)
    self.IDB = str(channelB)
    self.eventsA = []
    self.eventsB = []
    self.total_time = 0

  def add_event(self, edges):
    rea = edges["RE" + self.IDA]
    reb = edges["RE" + self.IDB]
    fea = edges["FE" + self.IDA]
    feb = edges["FE" + self.IDB]

    if rea.valid:
      self.eventsA.append((rea.t, 1))
    if reb.valid:
      self.eventsB.append((reb.t, 1))

    if fea.valid:
      self.eventsA.append((fea.t, -1))
    if feb.valid:
      self.eventsB.append((feb.t, -1))

    self.total_time = max(self.total_time, rea.t, reb.t, fea.t, feb.t)

  # Converts the timing information into a series of pulses (start/end pairs)
  def calculate_pulses(self):
    self.pulsesA = []
    self.pulsesB = []
    # Also make some nice waves for displaying data
    self.pulsesA_waveform = []
    self.pulsesB_waveform = []

    # Ensure we start pulse counting on a leading edge
    while self.eventsA[0][1] == -1:
      self.eventsA.pop(0)
    while self.eventsB[0][1] == -1:
      self.eventsB.pop(0)

    for i in range(1, len(self.eventsA), 2):
      # Check for corruption / something strange happening
      # (e.g. 2 successive rising edges with no falling edge in-between)
      while not (self.eventsA[i-1][1] == 1 and self.eventsA[i][1] == -1):
        i += 1

      # Append the edge list with the interval
      t1 = self.eventsA[i-1][0]
      t2 = self.eventsA[i][0]
      self.pulsesA.append([t1, t2])
      self.pulsesA_waveform.extend([[t1, 0], [t1, 1], [t2, 1], [t2, 0]])

    for i in range(1, len(self.eventsB), 2):
      while not (self.eventsB[i-1][1] == 1 and self.eventsB[i][1] == -1):
        i += 1
        if i == len(self.eventsB)-1:
          break

      t1 = self.eventsB[i-1][0]
      t2 = self.eventsB[i][0]
      self.pulsesB.append([t1, t2])
      self.pulsesB_waveform.extend([[t1, 0], [t1, 1], [t2, 1], [t2, 0]])

    self.pulsesA = np.array(self.pulsesA)
    self.pulsesB = np.array(self.pulsesB)
    self.pulsesA_waveform = np.array(self.pulsesA_waveform)
    self.pulsesB_waveform = np.array(self.pulsesB_waveform)

def load_txt_file(path):
  print("Loading " + path)
  clock_times = []
  tinyts = []
  data = Timeline(channelA=1, channelB=3)

  with open(path) as my_file:
    full_text = my_file.read()
    lines = full_text.split('\n')

    prev_raw_clock = -1
    clock_offset = 0

    # A regex pattern to match lines containing data
    pattern = '[0-9A-F]{8}\s[0-9A-F]{2}'
    for l in lines:
      this_line = l.strip()
      if re.match(pattern, this_line):
        # Cut out any junk at the beginning of each line
        this_line = this_line[re.search(pattern, this_line).start():]
        components = this_line.split(' ')[:15]

        # The clock resets every 103.07921508 seconds -- this code detects when
        # the clock jumps backwards and accounts for that difference
        # TODO: test/validate this
        raw_clock = float(int(components[0], 16))*CLOCK_SPEED
        if raw_clock < prev_raw_clock:
          clock_offset += float(int("FFFFFFFF", 16))*CLOCK_SPEED
        prev_raw_clock = raw_clock
        clock = raw_clock + clock_offset

        # Next, we've gotta look at the edges and figure out the
        # *exact* timing between triggers (more precise than the 24ns clock)
        # For reference, light moves about 1ft/ns, so 24ns is a while in
        # the muon timescale
        was_trigger = (int(components[1][0], 16) > 0) # False if follow-up
        labels = ["RE0", "FE0", "RE1", "FE1", "RE2", "FE2", "RE3", "FE3"]
        edges = {}
        for i in range(len(labels)):
          edges[labels[i]] = Edge(components[i+1], clock)
        data.add_event(edges)

    data.calculate_pulses()
  print("Loaded " + path + " with (" + str(len(data.eventsA)) + ", " + str(len(data.eventsB)) + ") events in (A, B) over " + str(data.total_time) + " seconds.")
  return data


# Note: TMC stands for "Time Measurement Chip"
def load_ht_file(path):
  # Clock    Rising/Falling Edges
  # 313C3401 00 01 01 23 00 01 00 01 00000000 0 0 0 0 8
  # 1        2  3  4  5  6  7  8  9  10       A B C D E

  # Word 1: 32-bit trigger count of the 24ns CPLD clock
  # Word 2: Rising Edge time count
  #   0-4 = TMC count of rising edge, res=24ns/32
  #   5   = 1 for valid rising edge, 0 for no edge
  #   6   = Always 0
  #   7   = 1 for new trigger, 0 for follow-up of a trigger
  # Word 3: Falling edge TMC count at input 0, Same as word 2 but no trigger tag
  # Word 4: Rising edge TMC count at input 1
  # Word 5: Falling edge TMC count at input 1
  # Word 6: RE2
  # Word 7: FE2
  # Word 8: RE3
  # Word 9: FE3
  # Word A...: GPS Stuff

  clock_times = []
  tinyts = []
  data = Timeline(channelA=1, channelB=3)

  with open(path, "rb") as my_file:
    full_text = my_file.read()
    full_text = full_text.replace(b'\x00', b'')
    lines = full_text.split(b'\r')

    prev_raw_clock = -1
    clock_offset = 0

    # A regex pattern to match lines containing data
    pattern = r'[0-9A-F]{8}\s[0-9A-F]{2}'
    for l in lines:
      this_line = l.decode("utf-8", errors="ignore").strip()
      if re.match(pattern, this_line):
        # Cut out any junk at the beginning of each line
        this_line = this_line[re.search(pattern, this_line).start():]
        components = this_line.split(' ')[:15]

        # The clock resets every 103.07921508 seconds -- this code detects when
        # the clock jumps backwards and accounts for that difference
        # TODO: test/validate this
        raw_clock = float(int(components[0], 16))*CLOCK_SPEED
        if raw_clock < prev_raw_clock:
          clock_offset += float(int("FFFFFFFF", 16))*CLOCK_SPEED
        prev_raw_clock = raw_clock
        clock = raw_clock + clock_offset

        # Next, we've gotta look at the edges and figure out the
        # *exact* timing between triggers (more precise than the 24ns clock)
        # For reference, light moves about 1ft/ns, so 24ns is a while in
        # the muon timescale
        was_trigger = (int(components[1][0], 16) > 0) # False if follow-up
        labels = ["RE0", "FE0", "RE1", "FE1", "RE2", "FE2", "RE3", "FE3"]
        edges = {}
        for i in range(len(labels)):
          edges[labels[i]] = Edge(components[i+1], clock)
        data.add_event(edges)

    data.calculate_pulses()
  return data

def transform_to_t_w(pulses, label):
  return [[a[0], a[1]-a[0], label] for a in pulses]

def get_reasonable_dts(data):
  data.pulsesA_t_w = transform_to_t_w(data.pulsesA, True)  # Record A (blue)
  data.pulsesB_t_w = transform_to_t_w(data.pulsesB, False) # Don't care 'bout B

  sorted_pulses_A = data.pulsesA_t_w
  sorted_pulses_A.sort(key=(lambda x : x[0]))
  sorted_pulses_A = np.array(sorted_pulses_A)[8:] # ignore starting few events
  dt_A = np.diff(sorted_pulses_A[:,0])

  sorted_pulses_B = data.pulsesB_t_w
  sorted_pulses_B.sort(key=(lambda x : x[0]))
  sorted_pulses_B = np.array(sorted_pulses_B)[8:] # ignore starting few events
  dt_B = np.diff(sorted_pulses_B[:,0])

  dt = np.concatenate((dt_A, dt_B))

  dt = dt[dt>0] # hence "reasonable"
  return dt


def get_log_hist(dat, bin_count):
  y, binEdges = np.histogram(np.log(dat), bins=bin_count)
  binCenters = 0.5*(binEdges[1:] + binEdges[:-1])
  binStd = np.sqrt(y)
  #plt.bar(binCenters, y, color='orange', width=0.2, yerr=binStd)
  #plt.yscale('log')
  #plt.show()
  return binCenters, y

angles = [1.5, 36, 56, 67.5, 77.5]
paths = ["data/telescope/TELESCOPE65.TXT",
         "data/telescope/TELESCOPE360.TXT",
         "data/telescope/TELESCOPE555.TXT",
         "data/telescope/TELESCOPE675.TXT",
         "data/telescope/TELESCOPE775.TXT"]

yA = []
yB = []
x = angles
for (a, path) in zip(angles, paths):
    d0 = load_txt_file(path)
    print("Detector A: " + str(a) + "° " + str(0.5*len(d0.eventsA)/d0.total_time))
    print("Detector B: " + str(a) + "° " + str(0.5*len(d0.eventsB)/d0.total_time))
    yA.append(0.5*len(d0.eventsA)/d0.total_time)
    yB.append(0.5*len(d0.eventsB)/d0.total_time)
plt.plot(x, yA, label="A")
plt.plot(x, yB, label="B")
yA = np.array(yA)
yB = np.array(yB)
yAB = (yA + yB)*0.5
plt.plot(x, yAB, label="Combined")
print(yAB)
plt.legend()
plt.grid()
plt.show()