#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct EngraveUniforms {
    noise_scale:  f32,
    octaves:      f32,   // cast to u32 in loop
    line_count:   f32,
    dot_density:  f32,
    dot_min_r:    f32,
    dot_max_r:    f32,
    _pad0:        f32,
    _pad1:        f32,
    bg_color:     vec4<f32>,
    dot_color:    vec4<f32>,
}

@group(2) @binding(0)
var<uniform> u: EngraveUniforms;

// ── Gradient noise ────────────────────────────────────────────────────────────

fn hash1(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

fn grad_dir(h: f32) -> vec2<f32> {
    let a = h * 6.28318530;
    return vec2(cos(a), sin(a));
}

fn grad_noise(p: vec2<f32>) -> f32 {
    let i  = floor(p);
    let f  = fract(p);
    let u2 = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let g00 = dot(grad_dir(hash1(i                )), f                );
    let g10 = dot(grad_dir(hash1(i + vec2(1., 0.) )), f - vec2(1., 0.) );
    let g01 = dot(grad_dir(hash1(i + vec2(0., 1.) )), f - vec2(0., 1.) );
    let g11 = dot(grad_dir(hash1(i + vec2(1., 1.) )), f - vec2(1., 1.) );
    return mix(mix(g00, g10, u2.x), mix(g01, g11, u2.x), u2.y) * 0.5 + 0.5;
}

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

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    // ── Dot grid cell ─────────────────────────────────────────────────────────
    let scaled  = uv * u.dot_density;
    let cell_id = floor(scaled);
    let local   = fract(scaled);                // (0,0)=bottom-left of cell

    // Sample height at dot center
    let center_uv = (cell_id + 0.5) / u.dot_density;
    let h         = fbm(center_uv * u.noise_scale);

    // Isoline proximity: 1 at isolines, 0 halfway between (triangle wave)
    let t         = fract(h * u.line_count);
    let proximity = 1.0 - abs(2.0 * t - 1.0);  // 0..1, peak at isoline

    // Map proximity → dot radius in [dot_min_r, dot_max_r]
    let dot_r = mix(u.dot_min_r, u.dot_max_r, proximity);

    // Antialiased circle test
    let dist  = length(local - 0.5);
    let fw    = fwidth(dist) * 0.5;
    let alpha = smoothstep(dot_r + fw, dot_r - fw, dist);

    return vec4(mix(u.bg_color.rgb, u.dot_color.rgb, alpha), 1.0);
}
