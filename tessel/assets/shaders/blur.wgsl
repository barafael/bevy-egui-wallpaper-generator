#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct BlurUniforms {
    radius: f32,
    inv_w:  f32,
    inv_h:  f32,
    _pad:   f32,
}

@group(2) @binding(0) var<uniform> u: BlurUniforms;
@group(2) @binding(1) var t: texture_2d<f32>;
@group(2) @binding(2) var s: sampler;

fn gauss(d: f32) -> f32 {
    return exp(-0.5 * d * d);
}

// 5×5 separable-weight Gaussian, step size = radius pixels.
// At radius=0 falls through to a plain texture fetch.
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if u.radius < 0.5 {
        return textureSample(t, s, in.uv);
    }

    let px = vec2(u.inv_w * u.radius, u.inv_h * u.radius);
    var col    = vec4(0.0);
    var weight = 0.0;

    for (var xi: i32 = -2; xi <= 2; xi++) {
        for (var yi: i32 = -2; yi <= 2; yi++) {
            let off = vec2(f32(xi), f32(yi));
            let w   = gauss(length(off));
            col    += textureSample(t, s, in.uv + off * px) * w;
            weight += w;
        }
    }

    return col / weight;
}
