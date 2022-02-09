#version 430

in vec3 in_vert;
in vec2 in_uv;
out vec2 v_coord;

void main()
{
    gl_Position = vec4(in_vert.xyz, 1.0);
    v_coord = in_uv;
}