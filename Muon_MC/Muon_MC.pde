PShader shadower;
ArrayList<String> lines = new ArrayList<String>();

int a(color c) {return (c >> 24) & 255; }
int r(color c) {return (c >> 16) & 255; }
int g(color c) {return (c >> 8) & 255;}
int b(color c) {return c & 255; }


//float SEP = 0.095;
//float SEP = 0.27;
float SEP = 0.0;
float PHI = PI * (sqrt(5) - 1);
float steradian_res = 0.001;
int TOTAL_SAMPLES;
int index = 1;
void setup() {
  size(512, 512, P2D);
  noStroke();
  shadower = loadShader("shadower.glsl");
  
  lines.add("theta,phi,areaA,areaB,areaI");
  
  shadower.set("RES",float(width),float(height));
  shadower.set("SEP",SEP);
  frameRate(500);
  TOTAL_SAMPLES = round(2*PI / steradian_res);
  println(TOTAL_SAMPLES);
}
boolean step() {
  if(index == TOTAL_SAMPLES) return true;
  float y = 1.0 - ((float)index / (TOTAL_SAMPLES - 1.f));
  float azimuth = (PHI * index)%TWO_PI;
  float zenith = acos(y);
  
  shadower.set("phi",azimuth);
  shadower.set("theta",zenith);
  shader(shadower);
  rect(0, 0, width, height);
  
  loadPixels();
  int total_A = 0;
  int total_B = 0;
  int total_AB = 0;
  int total = pixels.length;
  for(int c : pixels) {
    total_A += g(c);
    total_B += b(c);
    total_AB += r(c);
  }
  total_A /= 255;
  total_B /= 255;
  total_AB /= 255;
  
  float sf = 1.f / ((float)(width*height));
  
  float areaA = total_A*sf;
  float areaB = total_B*sf;
  float areaI = total_AB*sf;
  lines.add(zenith + "," + azimuth + "," + areaA + "," + areaB + "," + areaI);
  index++;
  return false;
}
void draw() {
  if(step()) {
    println(pixels.length);
    String[] o = new String[lines.size()];
    for(int i = 0; i < o.length; i++) {
      o[i] = lines.get(i);
    }
    saveStrings("light_MC_sep" + round(SEP*1000) + "mm.csv", o);
    noLoop();
  }
}
