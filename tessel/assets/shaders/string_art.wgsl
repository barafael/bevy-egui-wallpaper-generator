#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct StringArtUniforms {
    n_points:   f32,
    k_offset:   f32,
    line_width: f32,
    n_circles:  f32,
    center_a:   vec4<f32>,
    center_b:   vec4<f32>,
    fg_color:   vec4<f32>,
    bg_color:   vec4<f32>,
}

@group(2) @binding(0) var<uniform> u: StringArtUniforms;

fn sdf_seg(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    return length(ap - t * ab);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    let TAU = 6.28318530;

    let n_clamped = min(u32(u.n_points), 256u);
    let k = u32(u.k_offset);
    let n = n_clamped;

    var min_d = 1.0e9;

    for (var i = 0u; i < n_clamped; i++) {
        let a0 = f32(i) / f32(n) * TAU;
        let j = (i * k) % n;
        let a1 = f32(j) / f32(n) * TAU;

        var p0: vec2<f32>;
        var p1: vec2<f32>;

        if u.n_circles >= 2.0 {
            p0 = u.center_a.xy + u.center_a.z * vec2<f32>(cos(a0), sin(a0));
            p1 = u.center_b.xy + u.center_b.z * vec2<f32>(cos(a1), sin(a1));
        } else {
            p0 = u.center_a.xy + u.center_a.z * vec2<f32>(cos(a0), sin(a0));
            p1 = u.center_a.xy + u.center_a.z * vec2<f32>(cos(a1), sin(a1));
        }

        let d_seg = sdf_seg(uv, p0, p1);
        min_d = min(min_d, d_seg);
    }

    let fw = fwidth(min_d) * 0.5;
    let alpha = clamp(1.0 - smoothstep(u.line_width - fw, u.line_width + fw, min_d), 0.0, 1.0);

    let col = mix(u.bg_color.rgb, u.fg_color.rgb, alpha);
    return vec4<f32>(col, 1.0);
}
