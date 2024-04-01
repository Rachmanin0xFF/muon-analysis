import matplotlib.pyplot as plt
import numpy as np

# Adapted from http://jcgt.org/published/0007/03/04/
def boxIntersection(ro, rd, boxdims):
  invraydir = np.nan_to_num(1.0/np.array(rd))
  t0 = (-np.array(boxdims) - np.array(ro))*invraydir
  t1 = (np.array(boxdims) - np.array(ro))*invraydir
  tmin = np.minimum(t0, t1)
  tmax = np.maximum(t0, t1)
  return np.max(tmin) <= np.min(tmax)

def wedgeIntersection(ro, rd, wedgepos, wedgedims, mirror=[1.0, 1.0, 1.0]):
  P = np.prod(mirror)
  roo = (np.array(ro) - P*np.array(wedgepos))
  rdd = rd
  m = np.nan_to_num(1.0 / np.array(rdd))
  z = np.array([1.0 if a >= 0.0 else -1.0 for a in rdd])
  k = np.array(wedgedims)*z
  t1 = (-np.array(roo) - k)*m
  t2 = (-np.array(roo) + k)*m
  tN = np.max(t1)
  tF = np.min(t2)
  if tN > tF:
    return False

  k1 = wedgedims[1]*roo[0] - wedgedims[0]*roo[1]

  k2 = wedgedims[0]*rdd[1] - wedgedims[1]*rdd[0]
  if P < 0.0:
    k1 += 2.0*wedgedims[0]*roo[1]
    k2 -= 2.0*wedgedims[0]*rdd[1]
  tp = k1/k2

  if k1*P > tN*k2*P or (tp > tN and tp < tF):
    if P < 0.0:
      return False
    else:
      return True

  return P < 0.0

def trapIntersection(ro, rd, boxdims, truncation=0.8):
  width = boxdims[1]*0.5*truncation
  A = wedgeIntersection(ro, rd, [0, width + boxdims[1]*(1.0 - truncation), 0], [boxdims[0], width, boxdims[2]])
  B = wedgeIntersection(ro, rd, [0, width + boxdims[1]*(1.0 - truncation), 0], [boxdims[0], width, boxdims[2]], [1.0, -1.0, 1.0])
  C = boxIntersection(ro, rd, [boxdims[0], boxdims[1]*(1.0 - truncation), boxdims[2]])
  return (A or B) or C

def draw_box_outline(ax, x, y, z, w, l, h):
  xc = [x+w/2, x+w/2, x-w/2, x-w/2, x-w/2, x-w/2, x+w/2, x+w/2, x-w/2, x-w/2, x-w/2, x-w/2, x+w/2, x+w/2, x+w/2, x+w/2]
  yc = [y-l/2, y+l/2, y+l/2, y-l/2, y-l/2, y+l/2, y+l/2, y-l/2, y-l/2, y+l/2, y+l/2, y-l/2, y-l/2, y-l/2, y+l/2, y+l/2]
  zc = [z-h/2, z-h/2, z-h/2, z-h/2, z+h/2, z+h/2, z+h/2, z+h/2, z+h/2, z+h/2, z-h/2, z-h/2, z-h/2, z+h/2, z+h/2, z-h/2]

  ax.plot(xc, yc, zc)
  return np.sqrt(np.max(np.array(xc)**2 + np.array(yc)**2 + np.array(zc)**2))

def draw_trap_outline(ax, x, y, z, w, l, h, squishy):
  xc = [x+w/2, x+w/2, x-w/2, x-w/2, x-w/2, x-w/2, x+w/2, x+w/2, x-w/2, x-w/2, x-w/2, x-w/2, x+w/2, x+w/2, x+w/2, x+w/2]
  yc = [y-l/2, y+l/2, y+l/2, y-l/2, y-l/2, y+l/2, y+l/2, y-l/2, y-l/2, y+l/2, y+l/2, y-l/2, y-l/2, y-l/2, y+l/2, y+l/2]
  zc = [z-h/2, z-h/2, z-h/2, z-h/2, z+h/2, z+h/2, z+h/2, z+h/2, z+h/2, z+h/2, z-h/2, z-h/2, z-h/2, z+h/2, z+h/2, z-h/2]

  yc = np.array(yc)
  yc[np.array(xc) > x] = (np.array(yc[np.array(xc) > x]) - y)*squishy + np.array(y)
  ax.plot(xc, yc, zc)
  return np.sqrt(np.max(np.array(xc)**2 + np.array(yc)**2 + np.array(zc)**2))

def draw_ray(ax, ro, rd):
  t = np.arange(-10, 10, 0.1)
  xc = ro[0] + rd[0]*t
  yc = ro[1] + rd[1]*t
  zc = ro[2] + rd[2]*t
  ax.plot(xc, yc, zc)

# Working in units of meters

# "Heights" (longest axis)
DET_HEIGHT_BOX = 27.9/100.0 # 0.1cm
DET_HEIGHT_TRAP = 30.0/100.0 # 0.5cm
DET_HEIGHT_BOX2 = 14.2/100.0 # 0.1cm

# "Widths"
DET_WIDTH_BOX = 27.4/100.0 # 0.1cm
DET_WIDTH_BOX2 = 5.0/100.0 # 0.1cm

# Detector Thickness
DET_THICKNESS = 1.0/100.0

class Detector:
  def __init__(self, position, efficiency):
    self.r = position
  def intersects(self, ro, rd):
    boxsect = boxIntersection([ro[0] - self.r[0], ro[1] - self.r[1], ro[2] - self.r[2]], rd, [DET_HEIGHT_BOX*0.5, DET_WIDTH_BOX*0.5, DET_THICKNESS*0.5])
    trapsect = trapIntersection([ro[0] + (DET_HEIGHT_BOX + DET_HEIGHT_TRAP)/2.0 - self.r[0], ro[1] - self.r[1], ro[2] - self.r[2]], rd, [DET_HEIGHT_TRAP*0.5, DET_WIDTH_BOX*0.5, DET_THICKNESS*0.5], 1.0 - DET_WIDTH_BOX2/DET_WIDTH_BOX)
    return boxsect or trapsect
  def draw(self, ax=plt.figure().add_subplot(projection='3d')):
    self.ax = ax
    r1 = draw_box_outline(self.ax, self.r[0], self.r[1], self.r[2], DET_HEIGHT_BOX, DET_WIDTH_BOX, DET_THICKNESS)
    r2 = draw_trap_outline(self.ax, self.r[0] + (DET_HEIGHT_BOX + DET_HEIGHT_TRAP)/2.0, self.r[1], self.r[2], DET_HEIGHT_TRAP, DET_WIDTH_BOX, DET_THICKNESS, DET_WIDTH_BOX2/DET_WIDTH_BOX)
    self.ax.set_xlim(-0.5, 0.5)
    self.ax.set_ylim(-0.5, 0.5)
    self.ax.set_zlim(-0.5, 0.5)
    return max(r1, r2)
  def draw_diagnostic(self):
    # Test the trapezoid code
    arr = []
    for x in np.arange(-0.5, 0.5, 0.004):
      qr = []
      for y in np.arange(-0.5, 0.5, 0.004):
        B = 1.0 if self.intersects([0.01, 0.01, 1], [x, y, -1]) else 0.0
        qr.append(B)
      arr.append(qr)

    arr = np.array(arr)
    plt.imshow(arr, extent=[-0.5, 0.5, -0.5, 0.5])

my_ax = plt.figure().add_subplot(projection='3d')

SEP = 0.27
#SEP = 0.095

d1 = Detector([0.0, 0.0, -SEP*0.5], 0.8)
d2 = Detector([0.0, 0.0, SEP*0.5], 0.8)
system_radius = max(d1.draw(my_ax), d2.draw(my_ax))*1.1 # add a little padding

top_count = 0
bot_count = 0
coi_count = 0

data_arr = []

Nmax = 5000000
for i in range(0, Nmax):
  # generate random vector on sphere (Gaussian is spherically sym.)
  random_dir = np.array([np.random.normal(), np.random.normal(), np.random.normal()])
  random_dir *= 1.0 / np.sqrt(np.sum(random_dir**2))

  random_theta = np.random.rand()*2.0*np.pi
  # generate random radius and smoosh by Jacobian
  random_r = np.sqrt(np.random.rand())*system_radius

  # find an orthonormal basis for the space tangent to random vector on sphere
  tangent_space_u = np.cross(random_dir, [1, 1, 1])
  tangent_space_u *= 1.0 / np.sqrt(np.sum(tangent_space_u**2))
  tangent_space_v = np.cross(random_dir, tangent_space_u)

  u = np.cos(random_theta)*random_r
  v = np.sin(random_theta)*random_r

  # put our ray origin on the plane tangent to the sphere
  ro = random_dir + u*tangent_space_u + v*tangent_space_v
  # and ray direction is the negative vector going to the tangent point
  rd = -random_dir

  HITBOT = d1.intersects(ro, rd)
  HITTOP = d2.intersects(ro, rd)

  if HITBOT:
    bot_count += 1
  if HITTOP:
    top_count += 1
  if (HITBOT and HITTOP):
    coi_count += 1

  if bot_count%1000 == 0:
    print(bot_count, 100.0*i*1.0/Nmax)

  if HITBOT or HITTOP:
    data_arr.append([ro[0], ro[1], ro[2], rd[0], rd[1], rd[2], HITBOT, HITTOP])
print(bot_count, top_count, coi_count)
print(coi_count/(bot_count*0.5+top_count*0.5), "+/-", np.sqrt(coi_count)/(bot_count*0.5+top_count*0.5))
#draw_ray(my_ax, ro, rd)
data_arr = np.array(data_arr)

np.save("sep270mm.npy", data_arr)