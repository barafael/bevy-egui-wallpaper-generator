#import bevy_sprite::mesh2d_vertex_output::VertexOutput

// Crackle / ice-crack effect.
// Uses domain-warped Worley (cellular) noise to produce organic crack networks.
// F2-F1 is large inside cells, near-zero at Voronoi boundaries (crack centers).

struct CrackleUniforms {
    cell_count:   f32,   // Voronoi seed density
    crack_width:  f32,   // How wide the crack zone is (in Worley space)
    jitter:       f32,   // Voronoi seed jitter (0=grid, 1=fully random)
    warp:         f32,   // Domain warp strength (organic edge irregularity)
    color_var:    f32,   // Per-cell brightness variation
    inner_width:  f32,   // Inner vein width as fraction of crack_width
    inner_bright: f32,   // Inner vein brightness (0=dark, 1=same as cell)
    cell_depth:   f32,   // How much the corner gradient tints cell bodies
    bg_color:     vec4<f32>,   // Cell body base color
    crack_color:  vec4<f32>,   // Crack / vein fill color
    c_tl:         vec4<f32>,   // Corner gradient — top-left
    c_tr:         vec4<f32>,   // Corner gradient — top-right
    c_bl:         vec4<f32>,   // Corner gradient — bottom-left
    c_br:         vec4<f32>,   // Corner gradient — bottom-right
}

@group(2) @binding(0) var<uniform> u: CrackleUniforms;

fn hash11(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

fn hash2f(p: vec2<f32>) -> vec2<f32> {
    let a = dot(p, vec2(127.1, 311.7));
    let b = dot(p, vec2(269.5, 183.3));
    return fract(sin(vec2(a, b)) * 43758.5453);
}

// Smooth value noise — used for domain warping
fn val_noise(p: vec2<f32>) -> f32 {
    let i  = floor(p);
    let f  = fract(p);
    let s  = f * f * (3.0 - 2.0 * f);
    let a  = hash11(dot(i,                  vec2(1.0, 57.0)));
    let b  = hash11(dot(i + vec2(1.0, 0.0), vec2(1.0, 57.0)));
    let c  = hash11(dot(i + vec2(0.0, 1.0), vec2(1.0, 57.0)));
    let d  = hash11(dot(i + vec2(1.0, 1.0), vec2(1.0, 57.0)));
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

// Worley noise — returns (F1, F2, cell_hash)
// F1 = dist to nearest seed, F2 = dist to second-nearest
fn worley(p: vec2<f32>) -> vec3<f32> {
    let ip = floor(p);
    let fp = fract(p);
    var F1 = 8.0; var F2 = 8.0;
    var best: vec2<f32> = ip;
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let nb = ip + vec2<f32>(f32(dx), f32(dy));
            // Seed point: grid center ± jitter
            let r  = hash2f(nb);
            let pt = vec2<f32>(f32(dx), f32(dy)) + mix(vec2(0.5), r, u.jitter);
            let d  = length(fp - pt);
            if d < F1 { F2 = F1; F1 = d; best = nb; }
            else if d < F2 { F2 = d; }
        }
    }
    let h = hash11(dot(best, vec2(127.1, 311.7)));
    return vec3(F1, F2, h);
}

fn pastel(uv: vec2<f32>) -> vec3<f32> {
    let top = mix(u.c_tl.rgb, u.c_tr.rgb, uv.x);
    let bot = mix(u.c_bl.rgb, u.c_br.rgb, uv.x);
    return mix(top, bot, uv.y);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let sc = u.cell_count;

    // Domain warp: displace UV with smooth noise before computing Worley.
    // Divide by sc so the warp magnitude stays proportional to cell size.
    let wn = vec2<f32>(
        val_noise(uv * sc * 1.5),
        val_noise(uv * sc * 1.5 + vec2(5.2, 1.3)),
    ) * 2.0 - 1.0;
    let wuv = uv * sc + wn * u.warp;

    let wf        = worley(wuv);
    let F1        = wf.x;
    let F2        = wf.y;
    let cell_hash = wf.z;
    let crack_d   = F2 - F1;  // ~0 at crack center, larger inside cell

    // Cell body color: corner gradient + per-cell random variation
    let grad = pastel(uv);
    let var_ = (cell_hash * 2.0 - 1.0) * u.color_var;
    let cell_col = clamp(
        mix(u.bg_color.rgb, grad, u.cell_depth) + var_,
        vec3(0.0), vec3(1.0)
    );

    // Crack mask and inner vein mask
    let fw       = fwidth(crack_d);
    let crack_v  = 1.0 - smoothstep(u.crack_width - fw, u.crack_width + fw, crack_d);
    let inner_v  = 1.0 - smoothstep(
        u.crack_width * u.inner_width - fw,
        u.crack_width * u.inner_width + fw,
        crack_d
    );

    // The crack fill: dark base, slightly lighter at the very center (inner vein)
    let inner_col  = mix(u.crack_color.rgb, cell_col, 0.35 * u.inner_bright);
    let crack_col  = mix(u.crack_color.rgb, inner_col, inner_v);

    let out = mix(cell_col, crack_col, crack_v);
    return vec4(out, 1.0);
}
