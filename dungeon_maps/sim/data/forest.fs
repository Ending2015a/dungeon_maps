#version 430

#define EPS 0.0001
#define FAR 50.
#define NEAR 0.001

// Color encoding
//#define SHOW_NORMAL
//#define SHOW_TILES

// Object id
#define SKY 0.0
#define FLOOR 1.0
#define TRUNK 2.0
#define LEAF 3.0

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

float noise(vec2 p, float c)
{
    return abs(fract(sin(dot(p, vec2(12.331*p.x, 45.827*p.y)) * c + 4.152)*156.541));
}

mat3 rotY(float a)
{
    return mat3(
        cos(a), 0., -sin(a),
        0., 1., 0.,
        sin(a), 0., cos(a)
    );
}

mat3 rotZ(float a)
{
    return mat3(
        cos(a), -sin(a), 0.,
        sin(a), cos(a), 0.,
        0., 0., 1.
    );
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

Object sdPyramid(vec3 p, float sc, float h, float id)
{
    p /= sc;
    float m2 = h*h + 0.25;
    p.xz = abs(p.xz);
    p.xz = (p.z>p.x) ? p.zx : p.xz;
    p.xz -= 0.5;
    vec3 q = vec3( p.z, h*p.y - 0.5*p.x, h*p.x + 0.5*p.y);
    float s = max(-q.x,0.0);
    float t = clamp( (q.y-0.5*p.z)/(m2+0.25), 0.0, 1.0 );
    float a = m2*(q.x+s)*(q.x+s) + q.y*q.y;
    float b = m2*(q.x+0.5*t)*(q.x+0.5*t) + (q.y-m2*t)*(q.y-m2*t);
    float d2 = min(q.y,-q.x*m2-q.y*0.5) > 0.0 ? 0.0 : min(a,b);
    float d = sqrt( (d2+q.z*q.z)/m2 ) * sign(max(q.z,-p.y));
    return Object(d * sc, id);
}

Object sdTree(vec3 p, vec2 t)
{
    Object obj = Object(FAR, 0.0);
    vec4 rnd = vec4(
        noise(t, 0.223),
        noise(t, 4.549),
        noise(t, 7.157),
        noise(t, 9.5168));
    float sc = rnd.w * 0.2 + 0.7;
    p *= rotY(rnd.x*12.154);
    vec3 tp = p;
    float th = 0.;
    float h = 0.;
    tp.x += tp.y * rnd.x * 0.2 * (0.8 + 0.2 * rnd.w);
    tp.z += tp.y * rnd.z * 0.2 * (0.8 + 0.2 * rnd.w);
    th = 0.15;
    tp.y -= th;
    h += 2*th;
    obj = opUnion(obj, sdBox(tp, vec3(sc*0.1, th, sc*0.1), TRUNK));
    tp.y -= th;
    tp.x += -tp.y * rnd.x * 0.3 * (0.8 + 0.2 * rnd.w);
    tp.z += -tp.y * rnd.z * 0.3 * (0.8 + 0.2 * rnd.w);
    th = 0.2;
    tp.y -= th;
    h += 2*th;
    obj = opUnion(obj, sdBox(tp, vec3(sc*0.1, th, sc*0.1), TRUNK));
    tp = p;
    tp.y -= h-th;
    h += sc;
    obj = opUnion(obj, sdPyramid(tp*rotZ(rnd.z*0.1), sc*0.75, 1.3*sc, LEAF));
    tp.y -= sc*0.25;
    obj = opUnion(obj, sdPyramid(tp, sc*0.6, 1.6*sc, LEAF));
    tp.y -= sc*0.253;
    obj = opUnion(obj, sdPyramid(tp*rotZ(rnd.x*0.1), sc*0.5, 1.8*sc, LEAF));
    tp.y -= sc*0.26;
    obj = opUnion(obj, sdPyramid(tp, sc*0.4, 1.5*sc, LEAF));
    return obj;
}

Object sdSphere(vec3 p, float s, float id)
{
    return Object(length(p)-s, id);
}

Object sdPlane(vec3 p, vec3 n, float h, float id)
{
    return Object(dot(p, n) + h, id);
}

Object sdForest(vec3 p)
{
    p /= MAZE_SCALE;
    vec2 t = floor(p.xz);
    p.xz = fract(p.xz) - 0.5;
    vec2 offsets = vec2(
        fract(noise(t, 2.3) * 1.452),
        fract(noise(t, 6.54) * 3.679)
    ) * 0.4 - 0.2;
    p.xz += offsets;
    p *= MAZE_SCALE;
    float prob = fract(noise(t, 3.7) * 8.451) * 1.0;
    Object obj;
    if (prob < 0.7)
    {
        obj = sdTree(p, t);
    }
    else
    {
        obj = Object(FAR, 0);
    }
    return obj;
}

Object map(vec3 p)
{
    Object obj = Object(FAR, 0.0);
    obj = opUnion(obj, sdForest(p));
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

vec3 palette( in float t)
{
    return vec3(0.5) + vec3(0.5)*cos( 6.28318*(vec3(1.0)*t+vec3(0.0, 0.33, 0.67)) );
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
        if (hit.id == FLOOR)
            col = vec3(0.1, 0.4, 0.1);
        else if (hit.id == TRUNK)
            col = vec3(0.54, 0.27, 0.07);
        else if (hit.id == LEAF)
            col = vec3(0., 0.3, 0.);
        else
            col = palette(hit.id);
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