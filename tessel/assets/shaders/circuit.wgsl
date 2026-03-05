#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct CircuitUniforms {
    cell_size:   f32,
    trace_width: f32,
    via_radius:  f32,
    density:     f32,
    color_var:   f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    trace_color: vec4<f32>,
    via_color:   vec4<f32>,
    bg_color:    vec4<f32>,
}

@group(2) @binding(0) var<uniform> u: CircuitUniforms;

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    return fract(sin(vec2<f32>(
        dot(p, vec2<f32>(127.1, 311.7)),
        dot(p, vec2<f32>(269.5, 183.3))
    )) * 43758.5453123);
}

// Signed distance to a line segment from a to b, evaluated at point p.
fn sdf_seg(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let t  = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * t);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv   = in.uv;
    let suv  = uv * u.cell_size;
    let cell = floor(suv);
    let local = fract(suv);

    // Per-cell color variation
    let cell_hash = hash21(cell);
    let var_amount = (cell_hash * 2.0 - 1.0) * u.color_var;
    let trace_col = clamp(u.trace_color.rgb + var_amount, vec3<f32>(0.0), vec3<f32>(1.0));
    let via_col   = clamp(u.via_color.rgb + var_amount * 0.5, vec3<f32>(0.0), vec3<f32>(1.0));

    // Center of cell in local coords
    let center = vec2<f32>(0.5, 0.5);

    // Check which of the 4 half-segments from the cell center are active.
    // Each edge is identified by a point offset that is unique to that shared edge:
    //   right edge:  cell + (0.5, 0)   — shared with the cell to the right
    //   left edge:   cell - (0.5, 0)   = (cell.x-1, cell.y) + (0.5, 0)
    //   top edge:    cell + (0, 0.5)   — shared with the cell above
    //   bottom edge: cell - (0, 0.5)   = (cell.x, cell.y-1) + (0, 0.5)
    let has_right  = hash21(cell + vec2<f32>( 0.5,  0.0)) < u.density;
    let has_left   = hash21(cell + vec2<f32>(-0.5,  0.0)) < u.density;
    let has_top    = hash21(cell + vec2<f32>( 0.0,  0.5)) < u.density;
    let has_bottom = hash21(cell + vec2<f32>( 0.0, -0.5)) < u.density;

    // Count active connections for via rendering
    var n_connections = 0u;
    if has_right  { n_connections += 1u; }
    if has_left   { n_connections += 1u; }
    if has_top    { n_connections += 1u; }
    if has_bottom { n_connections += 1u; }

    // Half-segment endpoints in local cell coordinates
    let right_end  = vec2<f32>(1.0, 0.5);
    let left_end   = vec2<f32>(0.0, 0.5);
    let top_end    = vec2<f32>(0.5, 1.0);
    let bottom_end = vec2<f32>(0.5, 0.0);

    // Accumulate minimum SDF distance over all active half-segments
    var min_d = 1e9;

    if has_right {
        let d = sdf_seg(local, center, right_end);
        min_d = min(min_d, d);
    }
    if has_left {
        let d = sdf_seg(local, center, left_end);
        min_d = min(min_d, d);
    }
    if has_top {
        let d = sdf_seg(local, center, top_end);
        min_d = min(min_d, d);
    }
    if has_bottom {
        let d = sdf_seg(local, center, bottom_end);
        min_d = min(min_d, d);
    }

    // Via circle SDF at cell center
    let via_d = length(local - center) - u.via_radius;

    // Antialiased compositing
    let fw_trace = fwidth(min_d) * 0.5;
    let fw_via   = fwidth(via_d) * 0.5;

    let trace_alpha = select(
        0.0,
        1.0 - smoothstep(u.trace_width - fw_trace, u.trace_width + fw_trace, min_d),
        n_connections > 0u
    );
    let via_alpha = select(
        0.0,
        1.0 - smoothstep(-fw_via, fw_via, via_d),
        n_connections > 0u
    );

    var col = u.bg_color.rgb;
    col = mix(col, trace_col, trace_alpha);
    col = mix(col, via_col, via_alpha);

    return vec4<f32>(col, 1.0);
}
