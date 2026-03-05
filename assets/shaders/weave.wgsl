#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct WeaveUniforms {
    thread_count: f32,
    thread_width: f32,
    shadow:       f32,
    color_var:    f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    warp_color:   vec4<f32>,
    weft_color:   vec4<f32>,
    bg_color:     vec4<f32>,
}

@group(2) @binding(0) var<uniform> u: WeaveUniforms;

fn hash11(p: f32) -> f32 {
    var x = fract(p * 0.1031);
    x *= x + 33.33;
    x *= x + x;
    return fract(x);
}

fn hash12(p: vec2<f32>) -> f32 {
    var x = fract(p * 0.1031);
    x += dot(x, vec2<f32>(x.y + 33.33, x.x + 33.33));
    return fract((x.x + x.y) * x.x);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    let scaled_x = uv.x * u.thread_count;
    let scaled_y = uv.y * u.thread_count;

    let tx = fract(scaled_x);
    let ty = fract(scaled_y);
    let cell_x = floor(scaled_x);
    let cell_y = floor(scaled_y);

    let hw = u.thread_width * 0.5;

    let fw_x = fwidth(scaled_x) * 0.5;
    let fw_y = fwidth(scaled_y) * 0.5;

    let in_warp = smoothstep(0.5 - hw - fw_x, 0.5 - hw + fw_x, tx)
                * (1.0 - smoothstep(0.5 + hw - fw_x, 0.5 + hw + fw_x, tx));

    let in_weft = smoothstep(0.5 - hw - fw_y, 0.5 - hw + fw_y, ty)
                * (1.0 - smoothstep(0.5 + hw - fw_y, 0.5 + hw + fw_y, ty));

    let is_warp_top = step(0.5, fract((cell_x + cell_y) * 0.5));

    let warp_var = (hash11(cell_x * 127.1) * 2.0 - 1.0) * u.color_var;
    let weft_var = (hash11(cell_y * 311.7) * 2.0 - 1.0) * u.color_var;

    let warp_col = clamp(u.warp_color.rgb + vec3<f32>(warp_var), vec3<f32>(0.0), vec3<f32>(1.0));
    let weft_col = clamp(u.weft_color.rgb + vec3<f32>(weft_var), vec3<f32>(0.0), vec3<f32>(1.0));

    var col = u.bg_color.rgb;

    // Weft layer (horizontal threads), shadowed when warp is on top in overlap
    let weft_shadow = 1.0 - u.shadow * in_warp * is_warp_top;
    col = mix(col, weft_col * weft_shadow, in_weft);

    // Warp layer on top when is_warp_top
    let warp_shadow = 1.0 - u.shadow * in_weft * (1.0 - is_warp_top);
    col = mix(col, warp_col * warp_shadow, in_warp * is_warp_top);

    // When weft is on top, overdraw the warp overlap
    col = mix(col, weft_col, in_weft * in_warp * (1.0 - is_warp_top));

    return vec4<f32>(col, 1.0);
}
