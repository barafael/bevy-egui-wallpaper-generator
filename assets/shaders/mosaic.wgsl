#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct MosaicUniforms {
    tile_count:  f32,
    grout_width: f32,
    bevel_width: f32,
    noise_scale: f32,
    color_var:   f32,
    ambient:     f32,
    diffuse:     f32,
    specular:    f32,
    shininess:   f32,
    light_dir:   vec4<f32>,
    grout_color: vec4<f32>,
    c_tl:        vec4<f32>,
    c_tr:        vec4<f32>,
    c_bl:        vec4<f32>,
    c_br:        vec4<f32>,
}

@group(2) @binding(0)
var<uniform> u: MosaicUniforms;

fn hash1(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

fn hash1b(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2(269.5, 183.3))) * 17143.2918);
}

fn grad_dir(h: f32) -> vec2<f32> {
    let a = h * 6.28318530;
    return vec2(cos(a), sin(a));
}

// Gradient noise: returns [0, 1]
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

fn bilinear(uv: vec2<f32>) -> vec3<f32> {
    let top = mix(u.c_tl.rgb, u.c_tr.rgb, uv.x);
    let bot = mix(u.c_bl.rgb, u.c_br.rgb, uv.x);
    return mix(top, bot, uv.y);
}

// Height of tile surface: 0 at edges, 1 at flat centre.
// With a small bevel_width the flat area dominates → normal stays (0,0,1) → uniform lit.
fn tile_height(inner: vec2<f32>) -> f32 {
    let d = min(min(inner.x, 1.0 - inner.x), min(inner.y, 1.0 - inner.y));
    return smoothstep(0.0, u.bevel_width, d);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    // ── Tile grid ─────────────────────────────────────────────────────────────
    let scaled  = uv * u.tile_count;
    let tile_id = floor(scaled);
    let local   = fract(scaled);

    // ── Grout lines ───────────────────────────────────────────────────────────
    let g = u.grout_width * 0.5;
    if local.x < g || local.x > 1.0 - g || local.y < g || local.y > 1.0 - g {
        return u.grout_color;
    }

    // ── Remap to inner tile [0, 1] ────────────────────────────────────────────
    let inner = clamp((local - g) / (1.0 - 2.0 * g), vec2(0.0), vec2(1.0));

    // ── Bevel normal (narrow bevel → flat centre has normal (0,0,1)) ─────────
    let eps    = 0.01;
    let h      = tile_height(inner);
    let hx     = tile_height(clamp(inner + vec2(eps, 0.0), vec2(0.0), vec2(1.0)));
    let hy     = tile_height(clamp(inner + vec2(0.0, eps), vec2(0.0), vec2(1.0)));
    let bscale = 2.5;
    let normal = normalize(vec3(
        -(hx - h) / eps * bscale,
        -(hy - h) / eps * bscale,
        1.0,
    ));

    // ── Phong shading ─────────────────────────────────────────────────────────
    let light = normalize(u.light_dir.xyz);
    let view  = vec3(0.0, 0.0, 1.0);
    let diff  = max(0.0, dot(normal, light));
    let refl  = reflect(-light, normal);
    let spec  = pow(max(0.0, dot(refl, view)), u.shininess);
    let lit   = clamp(u.ambient + u.diffuse * diff + u.specular * spec, 0.0, 1.5);

    // ── Tile color ────────────────────────────────────────────────────────────
    // Use 2-D domain warp (sampled at tile centre) to create organic colour
    // regions without bleeding between tiles.  Each tile is a single flat colour.
    let tile_uv = (tile_id + 0.5) / u.tile_count;
    let wx      = grad_noise(tile_uv * u.noise_scale + vec2(0.0, 3.7)) - 0.5;
    let wy      = grad_noise(tile_uv * u.noise_scale + vec2(2.1, 0.0)) - 0.5;
    let color_uv = clamp(tile_uv + vec2(wx, wy) * 0.5, vec2(0.0), vec2(1.0));
    let base     = bilinear(color_uv);

    // Independent per-channel hash jitter for clear tile-to-tile contrast
    let rx = (hash1(tile_id)                  * 2.0 - 1.0) * u.color_var;
    let ry = (hash1b(tile_id)                 * 2.0 - 1.0) * u.color_var;
    let rz = (hash1(tile_id + vec2(7.3, 0.0)) * 2.0 - 1.0) * u.color_var;
    let color = clamp(base + vec3(rx, ry, rz), vec3(0.0), vec3(1.0));

    return vec4(color * lit, 1.0);
}
