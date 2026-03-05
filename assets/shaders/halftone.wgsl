#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct HalftoneUniforms {
    tile_freq:   f32,
    dot_density: f32,
    exponent:    f32,
    ambient:     f32,
    diffuse:     f32,
    specular:    f32,
    shininess:   f32,
    color_var:   f32,
    light_dir:   vec4<f32>,
    c_tl:        vec4<f32>,
    c_tr:        vec4<f32>,
    c_bl:        vec4<f32>,
    c_br:        vec4<f32>,
}

@group(2) @binding(0)
var<uniform> u: HalftoneUniforms;

fn surface_height(uv: vec2<f32>) -> f32 {
    let pi2 = 6.28318530;
    let h = (cos(uv.x * u.tile_freq * pi2) * cos(uv.y * u.tile_freq * pi2) + 1.0) * 0.5;
    return pow(h, u.exponent);
}

// Bilinear interpolation of the four corner pastel colors.
// uv: (0,0) = top-left, (1,1) = bottom-right (standard UV, y-down).
fn pastel_color(uv: vec2<f32>) -> vec3<f32> {
    let top = mix(u.c_tl.rgb, u.c_tr.rgb, uv.x);
    let bot = mix(u.c_bl.rgb, u.c_br.rgb, uv.x);
    return mix(top, bot, uv.y);
}

// Simple hash → [0, 1) from a 2D cell index.
fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    // Surface normals via finite differences (UV space, y-down)
    let eps   = 0.001;
    let h     = surface_height(uv);
    let hx    = surface_height(uv + vec2(eps, 0.0));
    let hy    = surface_height(uv + vec2(0.0, eps));
    let scale = 0.4;
    let normal = normalize(vec3(
        -(hx - h) / eps * scale,
        -(hy - h) / eps * scale,
        1.0,
    ));

    // Phong shading
    let light     = normalize(u.light_dir.xyz);
    let view      = vec3(0.0, 0.0, 1.0);
    let diff      = max(0.0, dot(normal, light));
    let refl      = reflect(-light, normal);
    let spec      = pow(max(0.0, dot(refl, view)), u.shininess);
    let intensity = clamp(u.ambient + u.diffuse * diff + u.specular * spec, 0.0, 1.0);

    // Halftone dot grid
    let dot_uv = fract(uv * u.dot_density);
    let cell   = floor(uv * u.dot_density);
    let dist   = length(dot_uv - 0.5);
    let dot_r  = (1.0 - intensity) * 0.5;

    // Antialiased dot edge
    let fw    = fwidth(dist) * 0.5;
    let alpha = 1.0 - smoothstep(dot_r - fw, dot_r + fw, dist);

    // Pastel color sampled at cell center, with per-dot random nudge
    let cell_uv   = (cell + 0.5) / u.dot_density;
    let rnd       = (hash2(cell) * 2.0 - 1.0) * u.color_var;
    let dot_color = clamp(pastel_color(cell_uv) + rnd, vec3(0.0), vec3(1.0));

    // White background, colored dots
    return vec4(mix(vec3(1.0), dot_color, alpha), 1.0);
}
