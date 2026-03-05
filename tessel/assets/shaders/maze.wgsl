#import bevy_sprite::mesh2d_vertex_output::VertexOutput

// Wall bitmask per cell: bit0=top, bit1=right, bit2=bottom, bit3=left
// Packing: 8 cells per u32 (4 bits each), 4 u32s per vec4<u32>
// walls[32] -> 32×4×8 = 1024 cells (supports up to 32×32 grid)

struct MazeUniforms {
    grid_w:      u32,
    grid_h:      u32,
    line_width:  f32,
    _pad:        f32,
    wall_color:  vec4<f32>,
    floor_color: vec4<f32>,
    walls:       array<vec4<u32>, 32>,
}

@group(2) @binding(0) var<uniform> u: MazeUniforms;

fn get_walls(x: u32, y: u32) -> u32 {
    let idx     = y * u.grid_w + x;
    let u32_idx = idx / 8u;
    let bit_idx = (idx % 8u) * 4u;
    let vec_idx = u32_idx / 4u;
    let comp    = u32_idx % 4u;
    var packed: u32;
    switch comp {
        case 0u: { packed = u.walls[vec_idx].x; }
        case 1u: { packed = u.walls[vec_idx].y; }
        case 2u: { packed = u.walls[vec_idx].z; }
        case 3u, default: { packed = u.walls[vec_idx].w; }
    }
    return (packed >> bit_idx) & 0xfu;
}

// Returns all-walls (0xf) for out-of-bounds cells (treats grid edge as wall).
fn get_walls_safe(x: i32, y: i32) -> u32 {
    if x < 0 || y < 0 || u32(x) >= u.grid_w || u32(y) >= u.grid_h {
        return 0xfu;
    }
    return get_walls(u32(x), u32(y));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    let gx = u32(uv.x * f32(u.grid_w));
    let gy = u32(uv.y * f32(u.grid_h));
    let cx = i32(min(gx, u.grid_w - 1u));
    let cy = i32(min(gy, u.grid_h - 1u));

    // Local UV within the cell [0, 1]
    let cell_uv = vec2<f32>(
        fract(uv.x * f32(u.grid_w)),
        fract(uv.y * f32(u.grid_h)),
    );

    let walls    = get_walls_safe(cx,     cy    );
    let w_left   = get_walls_safe(cx - 1, cy    );
    let w_right  = get_walls_safe(cx + 1, cy    );
    let w_top    = get_walls_safe(cx,     cy - 1);
    let w_bottom = get_walls_safe(cx,     cy + 1);

    // ── Edge wall strips ──────────────────────────────────────────────────────
    var min_d = 1.0;
    if (walls & 1u) != 0u { min_d = min(min_d, cell_uv.y);        } // top
    if (walls & 2u) != 0u { min_d = min(min_d, 1.0 - cell_uv.x); } // right
    if (walls & 4u) != 0u { min_d = min(min_d, 1.0 - cell_uv.y); } // bottom
    if (walls & 8u) != 0u { min_d = min(min_d, cell_uv.x);        } // left

    // ── Corner-junction squares ───────────────────────────────────────────────
    // The pixel at a cell-boundary junction is assigned to exactly one cell by
    // u32(floor), so walls that belong to neighbouring cells can leave a 1-pixel
    // gap.  We close each gap by drawing a square SDF (L∞ norm) at every corner
    // whose junction has at least one wall radiating from it.
    //
    // For junction J shared by (cx,cy), (cx±1,cy), (cx,cy±1):
    //   wall from current → bit in `walls`
    //   wall from adjacent → corresponding bit in neighbour
    //   (each physical wall is stored in both adjacent cells, so the diagonal
    //    cell is never needed)
    //
    // Bit masks per cell that lead into each corner:
    //   TL (→ junction (cx,  cy  )): current top|left (9),  left top|right (3),  top  bot|left  (12)
    //   TR (→ junction (cx+1,cy  )): current top|right(3),  right top|left (9),  top  bot|right  (6)
    //   BL (→ junction (cx,  cy+1)): current bot|left (12), left  bot|right(6),  bot  top|left   (9)
    //   BR (→ junction (cx+1,cy+1)): current bot|right(6),  right bot|left (12), bot  top|right  (3)

    if (walls & 9u)  != 0u || (w_left & 3u)   != 0u || (w_top    & 12u) != 0u {
        min_d = min(min_d, max(cell_uv.x, cell_uv.y));           // TL corner
    }
    if (walls & 3u)  != 0u || (w_right & 9u)  != 0u || (w_top    & 6u)  != 0u {
        min_d = min(min_d, max(1.0 - cell_uv.x, cell_uv.y));    // TR corner
    }
    if (walls & 12u) != 0u || (w_left & 6u)   != 0u || (w_bottom & 9u)  != 0u {
        min_d = min(min_d, max(cell_uv.x, 1.0 - cell_uv.y));    // BL corner
    }
    if (walls & 6u)  != 0u || (w_right & 12u) != 0u || (w_bottom & 3u)  != 0u {
        min_d = min(min_d, max(1.0 - cell_uv.x, 1.0 - cell_uv.y)); // BR corner
    }

    let fw         = fwidth(cell_uv.x);
    let wall_alpha = 1.0 - smoothstep(u.line_width - fw, u.line_width + fw, min_d);

    let output = mix(u.floor_color.rgb, u.wall_color.rgb, wall_alpha);
    return vec4<f32>(output, 1.0);
}
