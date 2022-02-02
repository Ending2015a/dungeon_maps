#version 430

#define EPS 0.0001
#define FAR 50.
#define NEAR 0.001

// Color encoding
//#define SHOW_NORMAL
//#define SHOW_TILES

// Object id
#define SKY 0.0
#define WALL 1.0
#define FLOOR 2.0

uniform float MIN_DEPTH;
uniform float MAX_DEPTH;
uniform int RAY_ITER;
uniform float RAY_MULT;
uniform int SHADOW_ITER;
uniform float SHADOW_MAX_STEP;
// Maze settings
uniform float MAZE_SCALE;
uniform float WALL_HEIGHT;
uniform float WALL_WIDTH;
// inputs
uniform float iTime;
uniform vec2 iResolution;
uniform float iHFOV;
uniform vec3 iPosition;
uniform vec3 iTarget;
// in vec2 v_uv: screen space coordniate
in vec2 v_coord;
// out color
out vec4 out_color;

// Storing ray information
struct Object
{
    float d;
    float id;
};

struct RayHit
{
    vec3 pos; // hit position
    vec3 nor; // normal
    float d; // distance
    float steps; // number of steps
    float id; // object id
};

float noise(vec2 p)
{
    return abs(fract(sin(dot(p, vec2(12.331*p.x, 45.827*p.y)))*156.541*p.x));
}

Object opUnion(Object d1, Object d2)
{
    return d1.d < d2.d ? d1 : d2;
}

Object sdBox(vec3 p, vec3 b, float id)
{
    vec3 q = abs(p) - b;
    return Object(length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0), id);
}

Object sdPlane(vec3 p, vec3 n, float h, float id)
{
    return Object(dot(p, n) + h, id);
}

Object sdMaze(vec3 p, float id)
{
    vec2 t = floor(p.xz / MAZE_SCALE);
    p.xz = fract(p.xz / MAZE_SCALE) - 0.5; // tile position
    p.x *= 2.0*floor(fract(noise(t) * 8.153) * 1.8) - 1.0; // random flip x
    float d = abs(1./(2.*sqrt(2)) - abs(dot(p.xz, vec2(1.))/sqrt(2.)));
    //return Object(d-WALL_WIDTH/2.0, id);
    return Object(max((d * MAZE_SCALE) - WALL_WIDTH / 2.0, p.y - WALL_HEIGHT), id);
}

Object map(vec3 p)
{
    Object obj = Object(FAR, 0.0);
    obj = opUnion(obj, sdMaze(p, WALL));
    obj = opUnion(obj, sdPlane(p, vec3(0., 1., 0.), 0., FLOOR));
    return obj;
}

vec3 Normal(vec3 p)
{
    vec2 e = vec2(0.01, 0.);
    return normalize(vec3(
        map(p+e.xyy).d-map(p-e.xyy).d,
        map(p+e.yxy).d-map(p-e.yxy).d,
        map(p+e.yyx).d-map(p-e.yyx).d
    ));
}

RayHit Trace(in vec3 ro, in vec3 rd)
{
    Object obj = Object(0.0, 0.0);
    float t = NEAR;
    int i = 0;
    for(i=0;i<RAY_ITER;++i)
    {
        obj = map(ro+rd*t);
        if (abs(obj.d) < EPS || t > FAR)
            break;
        t += obj.d * RAY_MULT;
    }
    RayHit hit;
    hit.d = FAR;
    hit.id = SKY;
    if (t < FAR)
    {
        hit.pos = ro+rd*t;
        hit.d = t;
        hit.nor = Normal(hit.pos);
        hit.steps = i;
        hit.id = obj.id;
    }
    return hit;
}

float Shadow(in vec3 ro, in vec3 rd, float k)
{
    float res = 1.;
    float t = 0.;
    Object obj = Object(0.0, 0.0);
    for(int i=0;i<SHADOW_ITER;++i){
    	obj = map(ro+rd*t);
        res = min(res, k*obj.d/t);
        if(res < 0.02)
            return 0.02;
        t += clamp(res, 0.001, SHADOW_MAX_STEP);
    }
    return clamp(res, 0.02, 1.0);
}

vec3 Shade_sky(in vec3 rd, in vec3 sd, in vec3 sc)
{
    float sm = max(dot(rd, sd), 0.0);
    vec3 sky = mix(vec3(.0, .1, .4), vec3(.3, .6, .8), 1.-rd.y);
    sky = sky + sc * min(pow(sm, 1500.0)*5.0, 1.);
    sky = sky + sc * min(pow(sm, 10.0) * .6, 1.0);
    return sky;
}

vec3 s2c(vec3 p)
{
    return p.x * vec3(sin(p.y)*cos(p.z), cos(p.y), sin(p.y)*sin(p.z));
}

vec3 Shade(in vec3 ro, in vec3 rd, in RayHit hit)
{
    vec3 col = vec3(0.); // color
    vec3 sd = normalize(s2c(vec3(1., sin(iTime/8.)*1.04, 0.))); // sun direction
    vec3 sc = vec3(1.0, 0.9, 0.717); // sun color
#ifdef SHOW_NORMAL
    col = hit.nor * 0.5 + 0.5;
    col *= Shadow(hit.pos+0.001*hit.nor, sd, 8.0);
#else
    vec3 skycol = Shade_sky(rd, sd, sc);
    if (hit.id == SKY)
    {
        col = skycol;
    }
    else
    {
        if (hit.id == WALL)
            col = vec3(0.9, 0.1, 0.1);
        else
            col = vec3(0.1, 0.9, 0.1); //FLOOR
        // blinn-phong
        vec3 hal = normalize(sd-rd);
        vec3 ambc = vec3(0.1);
        float gloss = 32.0;
        float amb = 1.0;
        float sdw = Shadow(hit.pos+0.001*hit.nor, sd, 8.0);
        float dif = clamp(dot(sd, hit.nor), 0., 1.) * sdw;
        float spe = pow(clamp(dot(hit.nor, hal), 0., 1.), gloss) * dif;
        vec3 lin = vec3(0.);
        lin += ambc * (0.05 + 0.95 * amb);
        lin += sc * dif * 0.8;
        col *= lin;
        col = pow(col, vec3(0.7, 0.9, 1.0));
        col += spe * 0.8; //specular
    }
    float haze = pow(smoothstep(0., 1., 1.-hit.d/FAR), 0.15);
    col = mix(skycol, col, haze);
#endif
    return col;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // focal length
    float fx = 1 / tan(iHFOV/2);
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.x;
    vec3 ro = iPosition;
    vec3 ta = iTarget;
    vec3 cf = normalize(ta-ro); // forward
    vec3 cs = normalize(cross(cf, vec3(0,1,0))); // side
    vec3 cu = normalize(cross(cs, cf)); // up
    vec3 rd = normalize(uv.x*cs+uv.y*cu+fx*cf); // ray direction
    // ray march
    RayHit hit = Trace(ro, rd);
    vec3 col = Shade(ro, rd, hit);
    col = clamp(pow(col, vec3(0.4545)), 0., 1.);
    fragColor = vec4(col, 1.0);
    float dist = hit.d * dot(rd, cf); // calibrate
    gl_FragDepth = clamp((dist-MIN_DEPTH)/(MAX_DEPTH - MIN_DEPTH), 0., 1.-EPS);
}

void main()
{
    mainImage(out_color, v_coord.xy * iResolution.xy);
}