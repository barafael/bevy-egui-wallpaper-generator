#import bevy_sprite::mesh2d_vertex_output::VertexOutput

// Tile bitmask: bit0=top, bit1=right, bit2=bottom, bit3=left
// Packing: 8 tiles per u32 (4 bits each), 4 u32s per vec4<u32>
// tiles[32] → 32×4×8 = 1024 cells max (supports up to 32×32 grid)

struct WfcUniforms {
    grid_w:     u32,
    grid_h:     u32,
    line_width: f32,
    color_var:  f32,
    fg_color:   vec4<f32>,
    bg_color:   vec4<f32>,
    tiles:      array<vec4<u32>, 32>,
}

@group(2) @binding(0) var<uniform> u: WfcUniforms;

fn hash1(p: vec2<u32>) -> f32 {
    var h = p.x * 1664525u + p.y * 1013904223u;
    h ^= h >> 13u;
    h *= 1664525u;
    h ^= h >> 17u;
    return f32(h) / 4294967295.0;
}

fn get_tile(x: u32, y: u32) -> u32 {
    let idx     = y * u.grid_w + x;
    let u32_idx = idx / 8u;
    let bit_idx = (idx % 8u) * 4u;
    let vec_idx = u32_idx / 4u;
    let comp    = u32_idx % 4u;
    let v = u.tiles[vec_idx];
    var packed: u32;
    switch comp {
        case 0u: { packed = v.x; }
        case 1u: { packed = v.y; }
        case 2u: { packed = v.z; }
        default: { packed = v.w; }
    }
    return (packed >> bit_idx) & 0xfu;
}

// SDF distance to the "pipe" shape for this tile type.
// cell_uv: local UV within cell, (0,0)=top-left, (1,1)=bottom-right
fn tile_sdf(tile: u32, t: vec2<f32>) -> f32 {
    switch tile {
        case 5u:  { return abs(t.x - 0.5); }                                // V: top+bottom
        case 10u: { return abs(t.y - 0.5); }                                // H: right+left
        case 3u:  { return abs(length(t - vec2(1.0, 0.0)) - 0.5); }        // TR corner
        case 6u:  { return abs(length(t - vec2(1.0, 1.0)) - 0.5); }        // BR corner
        case 12u: { return abs(length(t - vec2(0.0, 1.0)) - 0.5); }        // BL corner
        case 9u:  { return abs(length(t - vec2(0.0, 0.0)) - 0.5); }        // TL corner
        case 15u: { return min(abs(t.x - 0.5), abs(t.y - 0.5)); }          // Cross
        case 7u:  { // T_TRB: top+right+bottom — V line + right-half H
            return min(abs(t.x - 0.5), select(999.0, abs(t.y - 0.5), t.x >= 0.5));
        }
        case 14u: { // T_RBL: right+bottom+left — H line + bottom-half V
            return min(abs(t.y - 0.5), select(999.0, abs(t.x - 0.5), t.y >= 0.5));
        }
        case 13u: { // T_TBL: top+bottom+left — V line + left-half H
            return min(abs(t.x - 0.5), select(999.0, abs(t.y - 0.5), t.x <= 0.5));
        }
        case 11u: { // T_TRL: top+right+left — H line + top-half V
            return min(abs(t.y - 0.5), select(999.0, abs(t.x - 0.5), t.y <= 0.5));
        }
        default: { return 1.0; }
    }
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    // Grid cell
    let gx = u32(uv.x * f32(u.grid_w));
    let gy = u32(uv.y * f32(u.grid_h));
    let cx = min(gx, u.grid_w - 1u);
    let cy = min(gy, u.grid_h - 1u);

    // Local UV within cell [0, 1]
    let cell_uv = vec2<f32>(
        uv.x * f32(u.grid_w) - f32(cx),
        uv.y * f32(u.grid_h) - f32(cy),
    );

    let tile  = get_tile(cx, cy);
    let d     = tile_sdf(tile, cell_uv);

    // Antialiased edge
    let fw    = fwidth(cell_uv.x);
    let alpha = 1.0 - smoothstep(u.line_width - fw, u.line_width + fw, d);

    // Per-cell color variation
    let h  = hash1(vec2<u32>(cx, cy));
    let fg = clamp(u.fg_color.rgb + (h * 2.0 - 1.0) * u.color_var, vec3(0.0), vec3(1.0));

    return vec4(mix(u.bg_color.rgb, fg, alpha), 1.0);
}
