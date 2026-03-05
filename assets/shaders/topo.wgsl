#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct TopoUniforms {
    noise_scale: f32,
    octaves:     f32,   // cast to u32 in loop
    line_count:  f32,
    line_width:  f32,
    ambient:     f32,
    diffuse:     f32,
    specular:    f32,
    shininess:   f32,
    light_dir:   vec4<f32>,
    bg_color:    vec4<f32>,
    line_color:  vec4<f32>,
}

@group(2) @binding(0)
var<uniform> u: TopoUniforms;

// --- Gradient (Perlin-style) noise ---

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Maps a hash value to a pseudo-random unit direction.
fn grad_dir(h: f32) -> vec2<f32> {
    let a = h * 6.28318530;
    return vec2(cos(a), sin(a));
}

// Gradient noise: pseudo-random unit gradient dotted with offset.
fn grad_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Quintic smoothstep
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let g00 = dot(grad_dir(hash2(i                    )), f                    );
    let g10 = dot(grad_dir(hash2(i + vec2(1.0, 0.0)  )), f - vec2(1.0, 0.0)  );
    let g01 = dot(grad_dir(hash2(i + vec2(0.0, 1.0)  )), f - vec2(0.0, 1.0)  );
    let g11 = dot(grad_dir(hash2(i + vec2(1.0, 1.0)  )), f - vec2(1.0, 1.0)  );

    return mix(mix(g00, g10, u.x), mix(g01, g11, u.x), u.y) * 0.5 + 0.5;
}

// fBM: sum of octaves with halving amplitude each time.
fn fbm(p: vec2<f32>) -> f32 {
    var val  = 0.0;
    var amp  = 0.5;
    var freq = 1.0;
    let n    = u32(u.octaves);
    for (var i = 0u; i < n; i++) {
        val  += grad_noise(p * freq) * amp;
        freq *= 2.0;
        amp  *= 0.5;
    }
    return val;
}

fn height(uv: vec2<f32>) -> f32 {
    return fbm(uv * u.noise_scale);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv  = in.uv;
    let eps = 1.0 / 512.0;   // ~sub-pixel step in UV space

    let h   = height(uv);
    let hx  = height(uv + vec2(eps, 0.0));
    let hy  = height(uv + vec2(0.0, eps));

    // Surface normal from gradient (scale controls bumpiness)
    let bump_scale = 3.0;
    let normal = normalize(vec3(
        -(hx - h) / eps * bump_scale,
        -(hy - h) / eps * bump_scale,
        1.0,
    ));

    // Phong shading
    let light     = normalize(u.light_dir.xyz);
    let view      = vec3(0.0, 0.0, 1.0);
    let diff      = max(0.0, dot(normal, light));
    let refl      = reflect(-light, normal);
    let spec      = pow(max(0.0, dot(refl, view)), u.shininess);
    let intensity = clamp(u.ambient + u.diffuse * diff + u.specular * spec, 0.0, 1.5);

    // Isoline: distance to nearest multiple of (1/line_count).
    // fwidth must be taken on the pre-fract value; fwidth(fract(x)) has
    // undefined (exploding) derivatives at integer wrap-around points.
    let scaled = h * u.line_count;
    let fw     = fwidth(scaled) * 0.5;
    let t      = fract(scaled);
    let dist   = min(t, 1.0 - t);                 // 0 = on a line
    let mask   = 1.0 - smoothstep(u.line_width - fw, u.line_width + fw, dist);

    // Apply lighting separately to background and line
    let lit_bg   = u.bg_color.rgb   * intensity;
    let lit_line = u.line_color.rgb * intensity;

    return vec4(mix(lit_bg, lit_line, mask), 1.0);
}
