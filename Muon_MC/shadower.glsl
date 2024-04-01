#define PROCESSING_COLOR_SHADER

uniform float TIME;
uniform vec2 RES;
uniform float phi;
uniform float theta;
uniform float SEP;

bool boxIntersection(in vec3 ro, in vec3 rd, in vec3 boxdims) {
  vec3 invraydir = 1.0/rd;
  vec3 t0 = (-boxdims - ro)*invraydir;
  vec3 t1 = (boxdims - ro)*invraydir;
  vec3 tmin = vec3(min(t0.x, t1.x), min(t0.y, t1.y), min(t0.z, t1.z));
  vec3 tmax = vec3(max(t0.x, t1.x), max(t0.y, t1.y), max(t0.z, t1.z));
  return max(max(tmin.x, tmin.y), tmin.z) <= min(min(tmax.x, tmax.y), tmax.z);
}

bool wedgeIntersection(in vec3 ro, in vec3 rd, in vec3 wedgepos, in vec3 wedgedims, in vec3 mirror) {
  float P = mirror.x*mirror.y*mirror.z;
  vec3 roo = ro - wedgepos*P;
  vec3 rdd = rd;
  vec3 m = vec3(1.0, 1.0, 1.0) / rdd;
  vec3 z = sign(rdd);
  vec3 k = wedgedims*z;
  vec3 t1 = (-roo - k)*m;
  vec3 t2 = (-roo + k)*m;
  float tN = max(t1.x, max(t1.y, t1.z));
  float tF = min(t2.x, min(t2.y, t2.z));
  if(tN > tF) {
    return false;
	}

  float k1 = wedgedims.y*roo.x - wedgedims.x*roo.y;
  float k2 = wedgedims.x*rdd.y - wedgedims.y*rdd.x;
  float tp = k1/k2;
  if(P < 0.0) {
    k1 = -wedgedims.y*roo.x - wedgedims.x*roo.y;
    k2 = -wedgedims.x*rdd.y - wedgedims.y*rdd.x;
	tp = -k1/k2;
  }
  if((tp > tN && tp < tF) && P < 0.0) return true;
  
  if(k1 > tN*k2*P || (tp > tN && tp < tF)) {
    return P > 0.0;
  }
  return P < 0.0;
}

bool trapIntersection(in vec3 ro, in vec3 rd, in vec3 boxdims, in float truncation) {
  float width = boxdims.y*0.5*truncation;
  bool A = wedgeIntersection(ro, rd, vec3(0.0, width + boxdims.y*(1.0 - truncation), 0.0), vec3(boxdims.x, width, boxdims.z), vec3(1.0));
  bool B = wedgeIntersection(ro, rd, vec3(0.0, width + boxdims.y*(1.0 - truncation), 0.0), vec3(boxdims.x, width, boxdims.z), vec3(1.0, -1.0, 1.0));
  bool C = boxIntersection(ro, rd, vec3(boxdims.x, boxdims.y*(1.0 - truncation), boxdims.z));
  
  return C || B || A;
}

// "Heights" (longest axis)
const float DET_HEIGHT_BOX = 27.9/100.0; // 0.1cm
const float DET_HEIGHT_TRAP = 30.0/100.0; // 0.5cm
const float DET_HEIGHT_BOX2 = 14.2/100.0; // 0.1cm

// "Widths"
const float DET_WIDTH_BOX = 27.4/100.0; // 0.1cm
const float DET_WIDTH_BOX2 = 5.0/100.0; // 0.1cm

// Detector Thickness
const float DET_THICKNESS = 1.0/100.0;

bool intersects(in vec3 ro, in vec3 rd, in vec3 self) {
  bool boxsect  =  boxIntersection(vec3(ro.x - self.x, ro.y - self.y, ro.z - self.z), rd, 
								   vec3(DET_HEIGHT_BOX*0.5, DET_WIDTH_BOX*0.5, DET_THICKNESS*0.5));
  bool trapsect = trapIntersection(vec3(ro.x + (DET_HEIGHT_BOX + DET_HEIGHT_TRAP)/2.0 - self.x, ro.y - self.y, ro.z - self.z), rd,
								   vec3(DET_HEIGHT_TRAP*0.5, DET_WIDTH_BOX*0.5, DET_THICKNESS*0.5), 1.0 - DET_WIDTH_BOX2/DET_WIDTH_BOX);
  return boxsect || trapsect;
}

void main( void ) {
	vec2 position = gl_FragCoord.xy / RES.xy - vec2(0.5, 0.5);
	float fov = 1.0;
	
	vec3 ro = vec3(cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta));
	vec3 rd = -ro;
	vec3 up = vec3(0.0, 0.0, 1.0);
	
	vec3 right = normalize(cross(rd, up));
	up = cross(right, rd);
	
	ro*= 10.0;
	ro += up*position.y*fov + right*position.x*fov;
	
	gl_FragColor = vec4(sin(TIME), 1.0, 0.0, 1.0);

	bool detA = intersects(ro, rd, vec3(0.0, 0.0, -SEP*0.5));
	bool detB = intersects(ro, rd, vec3(0.0, 0.0,  SEP*0.5));
	
	vec3 cc = vec3(detA && detB, detA, detB);
	gl_FragColor = vec4(cc, 1.0);
}