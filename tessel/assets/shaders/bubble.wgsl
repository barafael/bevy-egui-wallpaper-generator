#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct BubbleUniforms {
    density:   f32,   // Voronoi seed density
    min_r:     f32,   // minimum bubble radius (fraction of cell size)
    max_r:     f32,   // maximum bubble radius
    border:    f32,   // border/gap thickness between bubbles
    jitter:    f32,   // seed jitter (0=grid, 1=random)
    color_var: f32,   // per-bubble color variation
    _pad0:     f32,
    _pad1:     f32,
    c_tl:      vec4<f32>,
    c_tr:      vec4<f32>,
    c_bl:      vec4<f32>,
    c_br:      vec4<f32>,
}

@group(2) @binding(0) var<uniform> u: BubbleUniforms;

// --- Hash functions ---

fn hash11(p: f32) -> f32 {
    var x = p;
    x = fract(x * 0.1031);
    x *= x + 33.33;
    x *= x + x;
    return fract(x);
}

fn hash2f(p: vec2<f32>) -> vec2<f32> {
    var q = vec2<f32>(
        dot(p, vec2<f32>(127.1, 311.7)),
        dot(p, vec2<f32>(269.5, 183.3)),
    );
    return fract(sin(q) * 43758.5453123);
}

// --- Worley noise ---
// Returns vec3(F1, F2, cell_hash) where distances are in Worley-space units.

fn worley(p: vec2<f32>) -> vec3<f32> {
    let i = floor(p);
    let f = fract(p);

    var F1        = 8.0;
    var F2        = 8.0;
    var cell_hash = 0.0;

    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let neighbor = vec2<f32>(f32(dx), f32(dy));
            let cell     = i + neighbor;

            let jitter_offset = hash2f(cell) * u.jitter + (1.0 - u.jitter) * vec2<f32>(0.5);
            let seed          = cell + jitter_offset;
            let diff          = seed - p;
            let dist          = length(diff);

            let h = hash11(dot(cell, vec2<f32>(127.1, 311.7)));

            if dist < F1 {
                F2        = F1;
                F1        = dist;
                cell_hash = h;
            } else if dist < F2 {
                F2 = dist;
            }
        }
    }

    return vec3<f32>(F1, F2, cell_hash);
}

// --- Pastel corner gradient ---

fn pastel(uv: vec2<f32>) -> vec3<f32> {
    let top    = mix(u.c_tl.rgb, u.c_tr.rgb, uv.x);
    let bottom = mix(u.c_bl.rgb, u.c_br.rgb, uv.x);
    return mix(top, bottom, uv.y);
}

// --- Fragment ---

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let suv = in.uv * u.density;
    let w   = worley(suv);

    let F1        = w.x;
    let cell_hash = w.z;

    // Per-bubble radius in Worley space (relative to density)
    let r = mix(u.min_r, u.max_r, cell_hash);

    let fw           = fwidth(F1);
    let bubble_alpha = 1.0 - smoothstep(r - u.border - fw, r - u.border + fw, F1);

    let base_col   = pastel(in.uv);
    let variation  = (cell_hash * 2.0 - 1.0) * u.color_var;
    let bubble_col = clamp(base_col + variation, vec3<f32>(0.0), vec3<f32>(1.0));
    let bg_col     = clamp(base_col - 0.15,      vec3<f32>(0.0), vec3<f32>(1.0));

    let output = mix(bg_col, bubble_col, bubble_alpha);
    return vec4<f32>(output, 1.0);
}
