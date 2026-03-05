#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct RdUniforms {
    scale:     f32,
    sharpness: f32,
    warp:      f32,
    balance:   f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    fg_color:  vec4<f32>,
    bg_color:  vec4<f32>,
    c_tl:      vec4<f32>,
    c_tr:      vec4<f32>,
    c_bl:      vec4<f32>,
    c_br:      vec4<f32>,
}

@group(2) @binding(0) var<uniform> u: RdUniforms;

fn hash11(p: f32) -> f32 {
    var x = fract(p * 0.1031);
    x *= x + 33.33;
    x *= x + x;
    return fract(x);
}

fn hash2f(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.13);
    p3 += dot(p3, p3.yzx + 3.333);
    return fract((p3.x + p3.y) * p3.z);
}

fn val_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash2f(i);
    let b = hash2f(i + vec2<f32>(1.0, 0.0));
    let c = hash2f(i + vec2<f32>(0.0, 1.0));
    let d = hash2f(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    let lacunarity = 2.0;
    let gain = 0.5;

    for (var i = 0; i < 5; i++) {
        value += amplitude * val_noise(p * frequency);
        frequency *= lacunarity;
        amplitude *= gain;
    }
    return value;
}

fn bilinear_corner_gradient(uv: vec2<f32>) -> vec3<f32> {
    let top    = mix(u.c_tl.rgb, u.c_tr.rgb, uv.x);
    let bottom = mix(u.c_bl.rgb, u.c_br.rgb, uv.x);
    return mix(bottom, top, uv.y);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    // Domain warp
    let wn = vec2<f32>(
        val_noise(uv * u.scale * 1.1) - 0.5,
        val_noise(uv * u.scale * 1.1 + vec2<f32>(3.7, 1.9)) - 0.5
    ) * u.warp;
    let wp = uv + wn;

    // Multi-scale noise for RD-like pattern
    let activator = fbm(wp * u.scale);
    let inhibitor  = fbm(wp * u.scale * 0.4);
    let signal     = activator - inhibitor * 0.7 + u.balance * 0.3;

    // Sigmoid-like contrast
    let inv_sharp = 1.0 / max(u.sharpness, 0.001);
    let mask = smoothstep(-inv_sharp, inv_sharp, signal);

    // Color
    let grad = bilinear_corner_gradient(uv);
    let fg   = mix(u.fg_color.rgb, grad, 0.4);
    let bg   = mix(u.bg_color.rgb, grad, 0.2);
    let col  = mix(bg, fg, mask);

    return vec4<f32>(col, 1.0);
}
