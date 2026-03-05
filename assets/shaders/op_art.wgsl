#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct OpArtUniforms {
    ring_count: f32,
    ring_width: f32,
    warp:       f32,
    freq:       f32,
    twist:      f32,
    _pad0:      f32,
    _pad1:      f32,
    _pad2:      f32,
    fg_color:   vec4<f32>,
    bg_color:   vec4<f32>,
}

@group(2) @binding(0) var<uniform> u: OpArtUniforms;

fn hash11(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let n = vec2<f32>(
        dot(p, vec2<f32>(127.1, 311.7)),
        dot(p, vec2<f32>(269.5, 183.3))
    );
    return fract(sin(n) * 43758.5453);
}

fn val_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);

    let a = hash22(i + vec2<f32>(0.0, 0.0)).x;
    let b = hash22(i + vec2<f32>(1.0, 0.0)).x;
    let c = hash22(i + vec2<f32>(0.0, 1.0)).x;
    let d = hash22(i + vec2<f32>(1.0, 1.0)).x;

    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    // Center to [-1, 1]
    var p = uv * 2.0 - vec2<f32>(1.0, 1.0);

    // Twist: rotate by twist * radius
    let angle = atan2(p.y, p.x) + u.twist * length(p);
    let r = length(p);
    let p_twisted = r * vec2<f32>(cos(angle), sin(angle));

    // Domain warp
    let wn_x = val_noise(uv * u.freq) - 0.5;
    let wn_y = val_noise(uv * u.freq + vec2<f32>(5.1, 2.3)) - 0.5;
    let wn = vec2<f32>(wn_x, wn_y) * u.warp;
    let p_warped = p_twisted + wn;

    // Concentric rings
    let dist = length(p_warped);
    let scaled = dist * u.ring_count;
    let t = fract(scaled);
    let fw = fwidth(scaled) * 0.5;

    // Ring: filled when ring_width -> 1, thin line when ring_width -> 0
    let on_ring = 1.0 - smoothstep(
        u.ring_width * 0.5 - fw,
        u.ring_width * 0.5 + fw,
        abs(t - 0.5)
    );

    let out_col = mix(u.bg_color.rgb, u.fg_color.rgb, on_ring);
    return vec4<f32>(out_col, 1.0);
}
