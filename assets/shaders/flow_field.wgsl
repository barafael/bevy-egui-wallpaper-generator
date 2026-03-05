#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct FlowFieldUniforms {
    freq:       f32,
    n_lines:    f32,
    line_width: f32,
    warp:       f32,
    octaves:    f32,
    color_var:  f32,
    _pad0:      f32,
    _pad1:      f32,
    c_tl:       vec4<f32>,
    c_tr:       vec4<f32>,
    c_bl:       vec4<f32>,
    c_br:       vec4<f32>,
}

@group(2) @binding(0) var<uniform> u: FlowFieldUniforms;

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
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash22(i + vec2<f32>(0.0, 0.0)).x;
    let b = hash22(i + vec2<f32>(1.0, 0.0)).x;
    let c = hash22(i + vec2<f32>(0.0, 1.0)).x;
    let d = hash22(i + vec2<f32>(1.0, 1.0)).x;

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p_in: vec2<f32>) -> f32 {
    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var frequency: f32 = 1.0;
    var p: vec2<f32> = p_in;
    let oct = clamp(round(u.octaves), 1.0, 8.0);
    for (var i: i32 = 0; i < 8; i++) {
        if f32(i) >= oct { break; }
        value += amplitude * val_noise(p * frequency);
        frequency *= 2.1;
        amplitude *= 0.45;
    }
    return value;
}

fn pastel(uv: vec2<f32>) -> vec3<f32> {
    let top    = mix(u.c_tl.rgb, u.c_tr.rgb, uv.x);
    let bottom = mix(u.c_bl.rgb, u.c_br.rgb, uv.x);
    return mix(bottom, top, uv.y);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    // Domain warp
    let wn_x = val_noise(uv * u.freq * 1.3) - 0.5;
    let wn_y = val_noise(uv * u.freq * 1.3 + vec2<f32>(3.7, 1.9)) - 0.5;
    let wn = vec2<f32>(wn_x, wn_y) * u.warp;
    let warped_uv = uv + wn;

    // Stream value from fBm on warped uv
    let stream = fbm(warped_uv * u.freq);

    // Iso-line band
    let scaled = stream * u.n_lines;
    let t = fract(scaled);
    let fw = fwidth(scaled) * 0.5;
    let on_line = 1.0 - smoothstep(u.line_width - fw, u.line_width + fw, min(t, 1.0 - t));

    // Color
    let base_col = clamp(pastel(uv), vec3<f32>(0.0), vec3<f32>(1.0));
    let band_id = floor(scaled);
    let variation = (hash11(band_id) * 2.0 - 1.0) * u.color_var;
    let line_col = clamp(base_col + variation, vec3<f32>(0.0), vec3<f32>(1.0));
    let bg_col = clamp(base_col - 0.12, vec3<f32>(0.0), vec3<f32>(1.0));

    let out_col = mix(bg_col, line_col, on_line);
    return vec4<f32>(out_col, 1.0);
}
