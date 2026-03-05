use bevy::{
    asset::RenderAssetUsages,
    camera::{RenderTarget, visibility::RenderLayers},
    ecs::system::SystemParam,
    mesh::PrimitiveTopology,
    prelude::*,
    render::{
        render_resource::{
            AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat, TextureUsages,
        },
        view::screenshot::{Screenshot, save_to_disk},
    },
    shader::ShaderRef,
    sprite_render::{Material2d, Material2dPlugin},
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use rand::{Rng, SeedableRng, rngs::StdRng};

// ── Style / Tab ───────────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy, Default, Debug)]
enum Tab {
    #[default]
    LowPoly,
    Voronoi,
    Halftone,
    Topo,
    Mosaic,
    Engrave,
    Wfc,
    Crackle,
    FlowField,
    OpArt,
    Weave,
    StringArt,
    RD,
    Maze,
    Circuit,
    Bubble,
    Penrose,
    Output,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum Style {
    LowPoly,
    Voronoi,
    Halftone,
    Topo,
    Mosaic,
    Engrave,
    Wfc,
    Crackle,
    FlowField,
    OpArt,
    Weave,
    StringArt,
    RD,
    Maze,
    Circuit,
    Bubble,
    Penrose,
}

// ── Parameters ────────────────────────────────────────────────────────────────

#[derive(Resource)]
struct Params {
    tab: Tab,
    style: Style,
    seed: u64,
    out_w: u32,
    out_h: u32,
    blur_radius: f32,
    screenshot_pending: bool,

    // ── Low poly ─────────────────────────────────────────────────────────────
    lp_cols: usize,
    lp_rows: usize,
    lp_jitter: f32,
    lp_color_var: f32,
    lp_c_tl: [f32; 3],
    lp_c_tr: [f32; 3],
    lp_c_bl: [f32; 3],
    lp_c_br: [f32; 3],

    // ── Voronoi ──────────────────────────────────────────────────────────────
    vor_cols: usize,
    vor_rows: usize,
    vor_jitter: f32,
    vor_border: f32,
    vor_elevation: f32,
    vor_color_var: f32,
    vor_c_tl: [f32; 3],
    vor_c_tr: [f32; 3],
    vor_c_bl: [f32; 3],
    vor_c_br: [f32; 3],

    // ── Halftone ─────────────────────────────────────────────────────────────
    ht_tile_freq: f32,
    ht_dot_density: f32,
    ht_exponent: f32,
    ht_ambient: f32,
    ht_diffuse: f32,
    ht_specular: f32,
    ht_shininess: f32,
    ht_light_elev: f32, // 0..90°
    ht_light_az: f32,   // 0..360°, 0=right, 270=up (UV y-down)
    ht_color_var: f32,
    ht_c_tl: [f32; 3],
    ht_c_tr: [f32; 3],
    ht_c_bl: [f32; 3],
    ht_c_br: [f32; 3],

    // ── Topo ─────────────────────────────────────────────────────────────────
    tp_noise_scale: f32,
    tp_octaves: f32, // 1..8
    tp_line_count: f32,
    tp_line_width: f32,
    tp_ambient: f32,
    tp_diffuse: f32,
    tp_specular: f32,
    tp_shininess: f32,
    tp_light_elev: f32,
    tp_light_az: f32,
    tp_bg_color: [f32; 3],
    tp_line_color: [f32; 3],

    // ── Mosaic ───────────────────────────────────────────────────────────────
    ms_tile_count: f32,
    ms_grout_width: f32,
    ms_bevel_width: f32,
    ms_noise_scale: f32,
    ms_color_var: f32,
    ms_ambient: f32,
    ms_diffuse: f32,
    ms_specular: f32,
    ms_shininess: f32,
    ms_light_elev: f32,
    ms_light_az: f32,
    ms_grout_color: [f32; 3],
    ms_c_tl: [f32; 3],
    ms_c_tr: [f32; 3],
    ms_c_bl: [f32; 3],
    ms_c_br: [f32; 3],

    // ── Engrave ──────────────────────────────────────────────────────────────
    eg_noise_scale: f32,
    eg_octaves: f32,
    eg_line_count: f32,
    eg_dot_density: f32,
    eg_dot_min_r: f32,
    eg_dot_max_r: f32,
    eg_bg_color: [f32; 3],
    eg_dot_color: [f32; 3],

    // ── WFC ──────────────────────────────────────────────────────────────────
    wfc_grid_w: usize,   // 4..=32
    wfc_grid_h: usize,   // 4..=24
    wfc_line_width: f32, // 0.02..=0.45
    wfc_color_var: f32,  // 0.0..=0.3
    wfc_tile_set: u8,    // 0=2-connected, 1=+cross, 2=+T-junctions
    wfc_fg_color: [f32; 3],
    wfc_bg_color: [f32; 3],
    wfc_grid: Vec<u8>, // CPU-side grid (not a shader param)

    // ── Crackle ──────────────────────────────────────────────────────────────
    ck_cell_count: f32,   // 4..=40
    ck_crack_width: f32,  // 0.01..=0.3
    ck_jitter: f32,       // 0.0..=1.0
    ck_warp: f32,         // 0.0..=1.5
    ck_color_var: f32,    // 0.0..=0.3
    ck_inner_width: f32,  // 0.0..=0.5 (fraction of crack_width)
    ck_inner_bright: f32, // 0.0..=1.0
    ck_cell_depth: f32,   // 0.0..=1.0
    ck_bg_color: [f32; 3],
    ck_crack_color: [f32; 3],
    ck_c_tl: [f32; 3],
    ck_c_tr: [f32; 3],
    ck_c_bl: [f32; 3],
    ck_c_br: [f32; 3],

    // ── Flow Field ───────────────────────────────────────────────────────────
    ff_freq: f32,
    ff_n_lines: f32,
    ff_line_width: f32,
    ff_warp: f32,
    ff_octaves: f32,
    ff_color_var: f32,
    ff_c_tl: [f32; 3],
    ff_c_tr: [f32; 3],
    ff_c_bl: [f32; 3],
    ff_c_br: [f32; 3],

    // ── Op Art ───────────────────────────────────────────────────────────────
    oa_ring_count: f32,
    oa_ring_width: f32,
    oa_warp: f32,
    oa_freq: f32,
    oa_twist: f32,
    oa_fg_color: [f32; 3],
    oa_bg_color: [f32; 3],

    // ── Weave ────────────────────────────────────────────────────────────────
    wv_thread_count: f32,
    wv_thread_width: f32,
    wv_shadow: f32,
    wv_color_var: f32,
    wv_warp_color: [f32; 3],
    wv_weft_color: [f32; 3],
    wv_bg_color: [f32; 3],

    // ── String Art ───────────────────────────────────────────────────────────
    sa_n_points: f32,
    sa_k_offset: f32,
    sa_line_width: f32,
    sa_n_circles: f32, // 1.0 or 2.0
    sa_cx_a: f32,
    sa_cy_a: f32,
    sa_r_a: f32,
    sa_cx_b: f32,
    sa_cy_b: f32,
    sa_r_b: f32,
    sa_fg_color: [f32; 3],
    sa_bg_color: [f32; 3],

    // ── Reaction-Diffusion ────────────────────────────────────────────────────
    rd_scale: f32,
    rd_sharpness: f32,
    rd_warp: f32,
    rd_balance: f32,
    rd_fg_color: [f32; 3],
    rd_bg_color: [f32; 3],
    rd_c_tl: [f32; 3],
    rd_c_tr: [f32; 3],
    rd_c_bl: [f32; 3],
    rd_c_br: [f32; 3],

    // ── Maze ─────────────────────────────────────────────────────────────────
    mz_grid_w: usize,
    mz_grid_h: usize,
    mz_line_width: f32,
    mz_wall_color: [f32; 3],
    mz_floor_color: [f32; 3],
    mz_grid: Vec<u8>,

    // ── Circuit Board ─────────────────────────────────────────────────────────
    cb_cell_size: f32,
    cb_trace_width: f32,
    cb_via_radius: f32,
    cb_density: f32,
    cb_color_var: f32,
    cb_trace_color: [f32; 3],
    cb_via_color: [f32; 3],
    cb_bg_color: [f32; 3],

    // ── Bubble Pack ───────────────────────────────────────────────────────────
    bp_density: f32,
    bp_min_r: f32,
    bp_max_r: f32,
    bp_border: f32,
    bp_jitter: f32,
    bp_color_var: f32,
    bp_c_tl: [f32; 3],
    bp_c_tr: [f32; 3],
    bp_c_bl: [f32; 3],
    bp_c_br: [f32; 3],

    // ── Penrose ───────────────────────────────────────────────────────────────
    pe_iterations: usize,
    pe_color_var: f32,
    pe_color_a: [f32; 3],
    pe_color_b: [f32; 3],
}

impl Default for Params {
    fn default() -> Self {
        Self {
            tab: Tab::LowPoly,
            style: Style::LowPoly,
            seed: 42,
            out_w: 1280,
            out_h: 720,
            blur_radius: 0.0,
            screenshot_pending: false,

            lp_cols: 24,
            lp_rows: 14,
            lp_jitter: 0.75,
            lp_color_var: 0.08,
            lp_c_tl: [0.18, 0.78, 0.70],
            lp_c_tr: [0.85, 0.82, 0.32],
            lp_c_bl: [0.50, 0.72, 0.42],
            lp_c_br: [0.85, 0.25, 0.10],

            vor_cols: 14,
            vor_rows: 8,
            vor_jitter: 0.65,
            vor_border: 10.0,
            vor_elevation: 0.12,
            vor_color_var: 0.03,
            vor_c_tl: [0.14, 0.14, 0.17],
            vor_c_tr: [0.18, 0.18, 0.22],
            vor_c_bl: [0.08, 0.08, 0.10],
            vor_c_br: [0.13, 0.13, 0.16],

            ht_tile_freq: 4.0,
            ht_dot_density: 60.0,
            ht_exponent: 1.5,
            ht_ambient: 0.05,
            ht_diffuse: 0.85,
            ht_specular: 0.40,
            ht_shininess: 32.0,
            ht_light_elev: 60.0,
            ht_light_az: 270.0,
            ht_color_var: 0.04,
            ht_c_tl: [0.82, 0.77, 0.95],
            ht_c_tr: [0.72, 0.88, 0.96],
            ht_c_bl: [0.96, 0.79, 0.84],
            ht_c_br: [0.76, 0.92, 0.84],

            tp_noise_scale: 3.5,
            tp_octaves: 6.0,
            tp_line_count: 20.0,
            tp_line_width: 0.04,
            tp_ambient: 0.08,
            tp_diffuse: 0.80,
            tp_specular: 0.60,
            tp_shininess: 48.0,
            tp_light_elev: 55.0,
            tp_light_az: 225.0,
            tp_bg_color: [0.03, 0.03, 0.03],
            tp_line_color: [0.18, 0.18, 0.18],

            ms_tile_count: 28.0,
            ms_grout_width: 0.08,
            ms_bevel_width: 0.05, // narrow → flat centre, bevel only at edge
            ms_noise_scale: 2.5,
            ms_color_var: 0.14, // strong per-tile contrast
            ms_ambient: 0.65,   // high ambient → mostly flat-lit tiles
            ms_diffuse: 0.45,
            ms_specular: 0.15,
            ms_shininess: 16.0,
            ms_light_elev: 50.0,
            ms_light_az: 315.0,
            ms_grout_color: [0.04, 0.05, 0.10],
            ms_c_tl: [0.08, 0.12, 0.45],
            ms_c_tr: [0.10, 0.45, 0.40],
            ms_c_bl: [0.12, 0.20, 0.55],
            ms_c_br: [0.15, 0.50, 0.30],

            eg_noise_scale: 2.5,
            eg_octaves: 5.0,
            eg_line_count: 24.0,
            eg_dot_density: 120.0,
            eg_dot_min_r: 0.0,
            eg_dot_max_r: 0.45,
            eg_bg_color: [1.0, 1.0, 1.0],
            eg_dot_color: [0.0, 0.0, 0.0],

            wfc_grid_w: 16,
            wfc_grid_h: 10,
            wfc_line_width: 0.10,
            wfc_color_var: 0.05,
            wfc_tile_set: 0,
            wfc_fg_color: [0.15, 0.35, 0.65],
            wfc_bg_color: [0.96, 0.97, 0.98],
            wfc_grid: run_wfc(16, 10, wfc_tiles(0), 42),

            ck_cell_count: 12.0,
            ck_crack_width: 0.12,
            ck_jitter: 0.90,
            ck_warp: 0.60,
            ck_color_var: 0.03,
            ck_inner_width: 0.20,
            ck_inner_bright: 0.55,
            ck_cell_depth: 0.08,
            ck_bg_color: [0.93, 0.97, 0.99],
            ck_crack_color: [0.05, 0.18, 0.28],
            ck_c_tl: [0.93, 0.97, 0.99],
            ck_c_tr: [0.88, 0.95, 0.99],
            ck_c_bl: [0.90, 0.96, 0.99],
            ck_c_br: [0.86, 0.93, 0.98],

            ff_freq: 3.0,
            ff_n_lines: 12.0,
            ff_line_width: 0.15,
            ff_warp: 0.8,
            ff_octaves: 5.0,
            ff_color_var: 0.06,
            ff_c_tl: [0.20, 0.55, 0.82],
            ff_c_tr: [0.20, 0.72, 0.62],
            ff_c_bl: [0.40, 0.30, 0.70],
            ff_c_br: [0.10, 0.58, 0.78],

            oa_ring_count: 8.0,
            oa_ring_width: 0.45,
            oa_warp: 0.20,
            oa_freq: 2.5,
            oa_twist: 0.0,
            oa_fg_color: [0.08, 0.08, 0.10],
            oa_bg_color: [0.95, 0.95, 0.92],

            wv_thread_count: 14.0,
            wv_thread_width: 0.65,
            wv_shadow: 0.35,
            wv_color_var: 0.08,
            wv_warp_color: [0.65, 0.22, 0.18],
            wv_weft_color: [0.20, 0.40, 0.65],
            wv_bg_color: [0.15, 0.12, 0.10],

            sa_n_points: 120.0,
            sa_k_offset: 71.0,
            sa_line_width: 0.0025,
            sa_n_circles: 1.0,
            sa_cx_a: 0.5,
            sa_cy_a: 0.5,
            sa_r_a: 0.42,
            sa_cx_b: 0.5,
            sa_cy_b: 0.5,
            sa_r_b: 0.28,
            sa_fg_color: [0.15, 0.30, 0.60],
            sa_bg_color: [0.98, 0.97, 0.96],

            rd_scale: 3.5,
            rd_sharpness: 6.0,
            rd_warp: 0.5,
            rd_balance: 0.0,
            rd_fg_color: [0.10, 0.12, 0.18],
            rd_bg_color: [0.88, 0.90, 0.95],
            rd_c_tl: [0.75, 0.85, 0.95],
            rd_c_tr: [0.80, 0.80, 0.90],
            rd_c_bl: [0.70, 0.80, 0.92],
            rd_c_br: [0.78, 0.88, 0.98],

            mz_grid_w: 20,
            mz_grid_h: 12,
            mz_line_width: 0.08,
            mz_wall_color: [0.10, 0.10, 0.12],
            mz_floor_color: [0.95, 0.94, 0.90],
            mz_grid: run_maze(20, 12, 42),

            cb_cell_size: 14.0,
            cb_trace_width: 0.06,
            cb_via_radius: 0.12,
            cb_density: 0.55,
            cb_color_var: 0.05,
            cb_trace_color: [0.20, 0.75, 0.40],
            cb_via_color: [0.50, 0.90, 0.55],
            cb_bg_color: [0.04, 0.06, 0.04],

            bp_density: 8.0,
            bp_min_r: 0.20,
            bp_max_r: 0.42,
            bp_border: 0.04,
            bp_jitter: 0.85,
            bp_color_var: 0.10,
            bp_c_tl: [0.85, 0.92, 0.98],
            bp_c_tr: [0.80, 0.88, 0.96],
            bp_c_bl: [0.82, 0.90, 0.97],
            bp_c_br: [0.78, 0.86, 0.95],

            pe_iterations: 5,
            pe_color_var: 0.06,
            pe_color_a: [0.78, 0.60, 0.22],
            pe_color_b: [0.30, 0.48, 0.72],
        }
    }
}

impl Params {
    fn light_dir(elev_deg: f32, az_deg: f32) -> Vec4 {
        let elev = elev_deg.to_radians();
        let az = az_deg.to_radians();
        let dir = Vec3::new(elev.cos() * az.cos(), elev.cos() * az.sin(), elev.sin()).normalize();
        Vec4::new(dir.x, dir.y, dir.z, 0.0)
    }

    fn to_ht_uniforms(&self) -> HalftoneUniforms {
        let c = |v: [f32; 3]| Vec4::new(v[0], v[1], v[2], 1.0);
        HalftoneUniforms {
            tile_freq: self.ht_tile_freq,
            dot_density: self.ht_dot_density,
            exponent: self.ht_exponent,
            ambient: self.ht_ambient,
            diffuse: self.ht_diffuse,
            specular: self.ht_specular,
            shininess: self.ht_shininess,
            color_var: self.ht_color_var,
            light_dir: Self::light_dir(self.ht_light_elev, self.ht_light_az),
            c_tl: c(self.ht_c_tl),
            c_tr: c(self.ht_c_tr),
            c_bl: c(self.ht_c_bl),
            c_br: c(self.ht_c_br),
        }
    }

    fn to_topo_uniforms(&self) -> TopoUniforms {
        let c = |v: [f32; 3]| Vec4::new(v[0], v[1], v[2], 1.0);
        TopoUniforms {
            noise_scale: self.tp_noise_scale,
            octaves: self.tp_octaves,
            line_count: self.tp_line_count,
            line_width: self.tp_line_width,
            ambient: self.tp_ambient,
            diffuse: self.tp_diffuse,
            specular: self.tp_specular,
            shininess: self.tp_shininess,
            light_dir: Self::light_dir(self.tp_light_elev, self.tp_light_az),
            bg_color: c(self.tp_bg_color),
            line_color: c(self.tp_line_color),
        }
    }

    fn to_engrave_uniforms(&self) -> EngraveUniforms {
        let c = |v: [f32; 3]| Vec4::new(v[0], v[1], v[2], 1.0);
        EngraveUniforms {
            noise_scale: self.eg_noise_scale,
            octaves: self.eg_octaves,
            line_count: self.eg_line_count,
            dot_density: self.eg_dot_density,
            dot_min_r: self.eg_dot_min_r,
            dot_max_r: self.eg_dot_max_r,
            _pad0: 0.0,
            _pad1: 0.0,
            bg_color: c(self.eg_bg_color),
            dot_color: c(self.eg_dot_color),
        }
    }

    fn to_mosaic_uniforms(&self) -> MosaicUniforms {
        let c = |v: [f32; 3]| Vec4::new(v[0], v[1], v[2], 1.0);
        MosaicUniforms {
            tile_count: self.ms_tile_count,
            grout_width: self.ms_grout_width,
            bevel_width: self.ms_bevel_width,
            noise_scale: self.ms_noise_scale,
            color_var: self.ms_color_var,
            ambient: self.ms_ambient,
            diffuse: self.ms_diffuse,
            specular: self.ms_specular,
            shininess: self.ms_shininess,
            light_dir: Self::light_dir(self.ms_light_elev, self.ms_light_az),
            grout_color: c(self.ms_grout_color),
            c_tl: c(self.ms_c_tl),
            c_tr: c(self.ms_c_tr),
            c_bl: c(self.ms_c_bl),
            c_br: c(self.ms_c_br),
        }
    }

    fn to_wfc_uniforms(&self) -> WfcUniforms {
        let c = |v: [f32; 3]| Vec4::new(v[0], v[1], v[2], 1.0);
        // Pack grid: 8 tiles per u32 (4 bits each), 4 u32s per UVec4
        let mut tiles = [UVec4::ZERO; 32];
        for (i, &tile) in self.wfc_grid.iter().enumerate().take(1024) {
            let u32_idx = i / 8;
            let bit_idx = (i % 8) * 4;
            let vec_idx = u32_idx / 4;
            let comp = u32_idx % 4;
            let val = (tile as u32 & 0xf) << bit_idx;
            match comp {
                0 => tiles[vec_idx].x |= val,
                1 => tiles[vec_idx].y |= val,
                2 => tiles[vec_idx].z |= val,
                _ => tiles[vec_idx].w |= val,
            }
        }
        WfcUniforms {
            grid_w: self.wfc_grid_w as u32,
            grid_h: self.wfc_grid_h as u32,
            line_width: self.wfc_line_width,
            color_var: self.wfc_color_var,
            fg_color: c(self.wfc_fg_color),
            bg_color: c(self.wfc_bg_color),
            tiles,
        }
    }

    fn to_ff_uniforms(&self) -> FlowFieldUniforms {
        let c = |v: [f32; 3]| Vec4::new(v[0], v[1], v[2], 1.0);
        FlowFieldUniforms {
            freq: self.ff_freq,
            n_lines: self.ff_n_lines,
            line_width: self.ff_line_width,
            warp: self.ff_warp,
            octaves: self.ff_octaves,
            color_var: self.ff_color_var,
            _pad0: 0.0,
            _pad1: 0.0,
            c_tl: c(self.ff_c_tl),
            c_tr: c(self.ff_c_tr),
            c_bl: c(self.ff_c_bl),
            c_br: c(self.ff_c_br),
        }
    }

    fn to_oa_uniforms(&self) -> OpArtUniforms {
        let c = |v: [f32; 3]| Vec4::new(v[0], v[1], v[2], 1.0);
        OpArtUniforms {
            ring_count: self.oa_ring_count,
            ring_width: self.oa_ring_width,
            warp: self.oa_warp,
            freq: self.oa_freq,
            twist: self.oa_twist,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            fg_color: c(self.oa_fg_color),
            bg_color: c(self.oa_bg_color),
        }
    }

    fn to_wv_uniforms(&self) -> WeaveUniforms {
        let c = |v: [f32; 3]| Vec4::new(v[0], v[1], v[2], 1.0);
        WeaveUniforms {
            thread_count: self.wv_thread_count,
            thread_width: self.wv_thread_width,
            shadow: self.wv_shadow,
            color_var: self.wv_color_var,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
            warp_color: c(self.wv_warp_color),
            weft_color: c(self.wv_weft_color),
            bg_color: c(self.wv_bg_color),
        }
    }

    fn to_sa_uniforms(&self) -> StringArtUniforms {
        let c = |v: [f32; 3]| Vec4::new(v[0], v[1], v[2], 1.0);
        StringArtUniforms {
            n_points: self.sa_n_points,
            k_offset: self.sa_k_offset,
            line_width: self.sa_line_width,
            n_circles: self.sa_n_circles,
            center_a: Vec4::new(self.sa_cx_a, self.sa_cy_a, self.sa_r_a, 0.0),
            center_b: Vec4::new(self.sa_cx_b, self.sa_cy_b, self.sa_r_b, 0.0),
            fg_color: c(self.sa_fg_color),
            bg_color: c(self.sa_bg_color),
        }
    }

    fn to_rd_uniforms(&self) -> RdUniforms {
        let c = |v: [f32; 3]| Vec4::new(v[0], v[1], v[2], 1.0);
        RdUniforms {
            scale: self.rd_scale,
            sharpness: self.rd_sharpness,
            warp: self.rd_warp,
            balance: self.rd_balance,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
            fg_color: c(self.rd_fg_color),
            bg_color: c(self.rd_bg_color),
            c_tl: c(self.rd_c_tl),
            c_tr: c(self.rd_c_tr),
            c_bl: c(self.rd_c_bl),
            c_br: c(self.rd_c_br),
        }
    }

    fn to_mz_uniforms(&self) -> MazeUniforms {
        let c = |v: [f32; 3]| Vec4::new(v[0], v[1], v[2], 1.0);
        let mut walls = [UVec4::ZERO; 32];
        for (i, &w) in self.mz_grid.iter().enumerate().take(1024) {
            let u32_idx = i / 8;
            let bit_idx = (i % 8) * 4;
            let vec_idx = u32_idx / 4;
            let comp = u32_idx % 4;
            let val = (w as u32 & 0xf) << bit_idx;
            match comp {
                0 => walls[vec_idx].x |= val,
                1 => walls[vec_idx].y |= val,
                2 => walls[vec_idx].z |= val,
                _ => walls[vec_idx].w |= val,
            }
        }
        MazeUniforms {
            grid_w: self.mz_grid_w as u32,
            grid_h: self.mz_grid_h as u32,
            line_width: self.mz_line_width,
            _pad: 0.0,
            wall_color: c(self.mz_wall_color),
            floor_color: c(self.mz_floor_color),
            walls,
        }
    }

    fn to_cb_uniforms(&self) -> CircuitUniforms {
        let c = |v: [f32; 3]| Vec4::new(v[0], v[1], v[2], 1.0);
        CircuitUniforms {
            cell_size: self.cb_cell_size,
            trace_width: self.cb_trace_width,
            via_radius: self.cb_via_radius,
            density: self.cb_density,
            color_var: self.cb_color_var,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            trace_color: c(self.cb_trace_color),
            via_color: c(self.cb_via_color),
            bg_color: c(self.cb_bg_color),
        }
    }

    fn to_bp_uniforms(&self) -> BubbleUniforms {
        let c = |v: [f32; 3]| Vec4::new(v[0], v[1], v[2], 1.0);
        BubbleUniforms {
            density: self.bp_density,
            min_r: self.bp_min_r,
            max_r: self.bp_max_r,
            border: self.bp_border,
            jitter: self.bp_jitter,
            color_var: self.bp_color_var,
            _pad0: 0.0,
            _pad1: 0.0,
            c_tl: c(self.bp_c_tl),
            c_tr: c(self.bp_c_tr),
            c_bl: c(self.bp_c_bl),
            c_br: c(self.bp_c_br),
        }
    }

    fn to_crackle_uniforms(&self) -> CrackleUniforms {
        let c = |v: [f32; 3]| Vec4::new(v[0], v[1], v[2], 1.0);
        CrackleUniforms {
            cell_count: self.ck_cell_count,
            crack_width: self.ck_crack_width,
            jitter: self.ck_jitter,
            warp: self.ck_warp,
            color_var: self.ck_color_var,
            inner_width: self.ck_inner_width,
            inner_bright: self.ck_inner_bright,
            cell_depth: self.ck_cell_depth,
            bg_color: c(self.ck_bg_color),
            crack_color: c(self.ck_crack_color),
            c_tl: c(self.ck_c_tl),
            c_tr: c(self.ck_c_tr),
            c_bl: c(self.ck_c_bl),
            c_br: c(self.ck_c_br),
        }
    }
}

// ── Halftone material ─────────────────────────────────────────────────────────

#[derive(Clone, ShaderType)]
struct HalftoneUniforms {
    tile_freq: f32,
    dot_density: f32,
    exponent: f32,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    color_var: f32,
    light_dir: Vec4,
    c_tl: Vec4,
    c_tr: Vec4,
    c_bl: Vec4,
    c_br: Vec4,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct HalftoneMaterial {
    #[uniform(0)]
    uniforms: HalftoneUniforms,
}

impl Material2d for HalftoneMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/halftone.wgsl".into()
    }
}

#[derive(Resource)]
struct HalftoneMaterialHandle(Handle<HalftoneMaterial>);

#[derive(Resource)]
struct HalftoneMeshHandle(Handle<Mesh>);

// ── Topo material ─────────────────────────────────────────────────────────────

#[derive(Clone, ShaderType)]
struct TopoUniforms {
    noise_scale: f32,
    octaves: f32,
    line_count: f32,
    line_width: f32,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    light_dir: Vec4,
    bg_color: Vec4,
    line_color: Vec4,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct TopoMaterial {
    #[uniform(0)]
    uniforms: TopoUniforms,
}

impl Material2d for TopoMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/topo.wgsl".into()
    }
}

#[derive(Resource)]
struct TopoMaterialHandle(Handle<TopoMaterial>);

#[derive(Resource)]
struct TopoMeshHandle(Handle<Mesh>);

// ── Engrave material ──────────────────────────────────────────────────────────

#[derive(Clone, ShaderType)]
struct EngraveUniforms {
    noise_scale: f32,
    octaves: f32,
    line_count: f32,
    dot_density: f32,
    dot_min_r: f32,
    dot_max_r: f32,
    _pad0: f32,
    _pad1: f32,
    bg_color: Vec4,
    dot_color: Vec4,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct EngraveMaterial {
    #[uniform(0)]
    uniforms: EngraveUniforms,
}

impl Material2d for EngraveMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/engrave.wgsl".into()
    }
}

#[derive(Resource)]
struct EngraveMaterialHandle(Handle<EngraveMaterial>);

#[derive(Resource)]
struct EngraveMeshHandle(Handle<Mesh>);

// ── Mosaic material ───────────────────────────────────────────────────────────

#[derive(Clone, ShaderType)]
struct MosaicUniforms {
    tile_count: f32,
    grout_width: f32,
    bevel_width: f32,
    noise_scale: f32,
    color_var: f32,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    light_dir: Vec4,
    grout_color: Vec4,
    c_tl: Vec4,
    c_tr: Vec4,
    c_bl: Vec4,
    c_br: Vec4,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct MosaicMaterial {
    #[uniform(0)]
    uniforms: MosaicUniforms,
}

impl Material2d for MosaicMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/mosaic.wgsl".into()
    }
}

#[derive(Resource)]
struct MosaicMaterialHandle(Handle<MosaicMaterial>);

#[derive(Resource)]
struct MosaicMeshHandle(Handle<Mesh>);

// ── WFC material ──────────────────────────────────────────────────────────────
//
// Tile bitmask: bit0=top, bit1=right, bit2=bottom, bit3=left
// Packed: 8 tiles per u32 (4 bits each), 4 u32s per UVec4 → 32 tiles/UVec4
// [UVec4; 32] holds 1024 tiles (up to 32×32 grid)

#[derive(Clone, ShaderType)]
struct WfcUniforms {
    grid_w: u32,
    grid_h: u32,
    line_width: f32,
    color_var: f32,
    fg_color: Vec4,
    bg_color: Vec4,
    tiles: [UVec4; 32],
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct WfcMaterial {
    #[uniform(0)]
    uniforms: WfcUniforms,
}

impl Material2d for WfcMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/wfc.wgsl".into()
    }
}

#[derive(Resource)]
struct WfcMaterialHandle(Handle<WfcMaterial>);
#[derive(Resource)]
struct WfcMeshHandle(Handle<Mesh>);

// ── Crackle material ──────────────────────────────────────────────────────────

#[derive(Clone, ShaderType)]
struct CrackleUniforms {
    cell_count: f32,
    crack_width: f32,
    jitter: f32,
    warp: f32,
    color_var: f32,
    inner_width: f32,
    inner_bright: f32,
    cell_depth: f32,
    bg_color: Vec4,
    crack_color: Vec4,
    c_tl: Vec4,
    c_tr: Vec4,
    c_bl: Vec4,
    c_br: Vec4,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct CrackleMaterial {
    #[uniform(0)]
    uniforms: CrackleUniforms,
}

impl Material2d for CrackleMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/crackle.wgsl".into()
    }
}

#[derive(Resource)]
struct CrackleMaterialHandle(Handle<CrackleMaterial>);
#[derive(Resource)]
struct CrackleMeshHandle(Handle<Mesh>);

// ── Flow Field material ────────────────────────────────────────────────────────

#[derive(Clone, ShaderType)]
struct FlowFieldUniforms {
    freq: f32,
    n_lines: f32,
    line_width: f32,
    warp: f32,
    octaves: f32,
    color_var: f32,
    _pad0: f32,
    _pad1: f32,
    c_tl: Vec4,
    c_tr: Vec4,
    c_bl: Vec4,
    c_br: Vec4,
}
#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct FlowFieldMaterial {
    #[uniform(0)]
    uniforms: FlowFieldUniforms,
}
impl Material2d for FlowFieldMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/flow_field.wgsl".into()
    }
}
#[derive(Resource)]
struct FlowFieldMaterialHandle(Handle<FlowFieldMaterial>);
#[derive(Resource)]
struct FlowFieldMeshHandle(Handle<Mesh>);

// ── Op Art material ────────────────────────────────────────────────────────────

#[derive(Clone, ShaderType)]
struct OpArtUniforms {
    ring_count: f32,
    ring_width: f32,
    warp: f32,
    freq: f32,
    twist: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    fg_color: Vec4,
    bg_color: Vec4,
}
#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct OpArtMaterial {
    #[uniform(0)]
    uniforms: OpArtUniforms,
}
impl Material2d for OpArtMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/op_art.wgsl".into()
    }
}
#[derive(Resource)]
struct OpArtMaterialHandle(Handle<OpArtMaterial>);
#[derive(Resource)]
struct OpArtMeshHandle(Handle<Mesh>);

// ── Weave material ─────────────────────────────────────────────────────────────

#[derive(Clone, ShaderType)]
struct WeaveUniforms {
    thread_count: f32,
    thread_width: f32,
    shadow: f32,
    color_var: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    warp_color: Vec4,
    weft_color: Vec4,
    bg_color: Vec4,
}
#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct WeaveMaterial {
    #[uniform(0)]
    uniforms: WeaveUniforms,
}
impl Material2d for WeaveMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/weave.wgsl".into()
    }
}
#[derive(Resource)]
struct WeaveMaterialHandle(Handle<WeaveMaterial>);
#[derive(Resource)]
struct WeaveMeshHandle(Handle<Mesh>);

// ── String Art material ────────────────────────────────────────────────────────

#[derive(Clone, ShaderType)]
struct StringArtUniforms {
    n_points: f32,
    k_offset: f32,
    line_width: f32,
    n_circles: f32,
    center_a: Vec4,
    center_b: Vec4,
    fg_color: Vec4,
    bg_color: Vec4,
}
#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct StringArtMaterial {
    #[uniform(0)]
    uniforms: StringArtUniforms,
}
impl Material2d for StringArtMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/string_art.wgsl".into()
    }
}
#[derive(Resource)]
struct StringArtMaterialHandle(Handle<StringArtMaterial>);
#[derive(Resource)]
struct StringArtMeshHandle(Handle<Mesh>);

// ── Reaction-Diffusion material ────────────────────────────────────────────────

#[derive(Clone, ShaderType)]
struct RdUniforms {
    scale: f32,
    sharpness: f32,
    warp: f32,
    balance: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    fg_color: Vec4,
    bg_color: Vec4,
    c_tl: Vec4,
    c_tr: Vec4,
    c_bl: Vec4,
    c_br: Vec4,
}
#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct RdMaterial {
    #[uniform(0)]
    uniforms: RdUniforms,
}
impl Material2d for RdMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/reaction_diffusion.wgsl".into()
    }
}
#[derive(Resource)]
struct RdMaterialHandle(Handle<RdMaterial>);
#[derive(Resource)]
struct RdMeshHandle(Handle<Mesh>);

// ── Maze material ──────────────────────────────────────────────────────────────

#[derive(Clone, ShaderType)]
struct MazeUniforms {
    grid_w: u32,
    grid_h: u32,
    line_width: f32,
    _pad: f32,
    wall_color: Vec4,
    floor_color: Vec4,
    walls: [UVec4; 32],
}
#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct MazeMaterial {
    #[uniform(0)]
    uniforms: MazeUniforms,
}
impl Material2d for MazeMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/maze.wgsl".into()
    }
}
#[derive(Resource)]
struct MazeMaterialHandle(Handle<MazeMaterial>);
#[derive(Resource)]
struct MazeMeshHandle(Handle<Mesh>);

// ── Circuit Board material ─────────────────────────────────────────────────────

#[derive(Clone, ShaderType)]
struct CircuitUniforms {
    cell_size: f32,
    trace_width: f32,
    via_radius: f32,
    density: f32,
    color_var: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    trace_color: Vec4,
    via_color: Vec4,
    bg_color: Vec4,
}
#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct CircuitMaterial {
    #[uniform(0)]
    uniforms: CircuitUniforms,
}
impl Material2d for CircuitMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/circuit.wgsl".into()
    }
}
#[derive(Resource)]
struct CircuitMaterialHandle(Handle<CircuitMaterial>);
#[derive(Resource)]
struct CircuitMeshHandle(Handle<Mesh>);

// ── Bubble Pack material ───────────────────────────────────────────────────────

#[derive(Clone, ShaderType)]
struct BubbleUniforms {
    density: f32,
    min_r: f32,
    max_r: f32,
    border: f32,
    jitter: f32,
    color_var: f32,
    _pad0: f32,
    _pad1: f32,
    c_tl: Vec4,
    c_tr: Vec4,
    c_bl: Vec4,
    c_br: Vec4,
}
#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct BubbleMaterial {
    #[uniform(0)]
    uniforms: BubbleUniforms,
}
impl Material2d for BubbleMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/bubble.wgsl".into()
    }
}
#[derive(Resource)]
struct BubbleMaterialHandle(Handle<BubbleMaterial>);
#[derive(Resource)]
struct BubbleMeshHandle(Handle<Mesh>);

// ── Blur post-process material ─────────────────────────────────────────────────

#[derive(Clone, ShaderType)]
struct BlurUniforms {
    radius: f32,
    inv_w: f32,
    inv_h: f32,
    _pad: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct BlurMaterial {
    #[uniform(0)]
    uniforms: BlurUniforms,
    #[texture(1)]
    #[sampler(2)]
    source: Handle<Image>,
}

impl Material2d for BlurMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/blur.wgsl".into()
    }
}

#[derive(Resource)]
struct BlurMaterialHandle(Handle<BlurMaterial>);
#[derive(Resource)]
struct BlurMeshHandle(Handle<Mesh>);
/// The intermediate render target Camera A writes to.
#[derive(Resource)]
struct SceneRenderTarget(Handle<Image>);
/// Entity ID of Camera A (scene → render target), needed for resize.
#[derive(Resource)]
struct SceneCameraEntity(Entity);

/// Marker component on every WGSL-shader quad; carries the style it belongs to.
#[derive(Component, Clone, Copy)]
struct StyleId(Style);

// ── Bundled params (keeps ui() under the 16-param system limit) ──────────────

#[derive(SystemParam)]
struct ShaderHandles<'w> {
    ht_mat: Res<'w, HalftoneMaterialHandle>,
    ht_mesh: Res<'w, HalftoneMeshHandle>,
    tp_mat: Res<'w, TopoMaterialHandle>,
    tp_mesh: Res<'w, TopoMeshHandle>,
    ms_mat: Res<'w, MosaicMaterialHandle>,
    ms_mesh: Res<'w, MosaicMeshHandle>,
    eg_mat: Res<'w, EngraveMaterialHandle>,
    eg_mesh: Res<'w, EngraveMeshHandle>,
    wfc_mat: Res<'w, WfcMaterialHandle>,
    wfc_mesh: Res<'w, WfcMeshHandle>,
    ck_mat: Res<'w, CrackleMaterialHandle>,
    ck_mesh: Res<'w, CrackleMeshHandle>,
    ff_mat: Res<'w, FlowFieldMaterialHandle>,
    ff_mesh: Res<'w, FlowFieldMeshHandle>,
    oa_mat: Res<'w, OpArtMaterialHandle>,
    oa_mesh: Res<'w, OpArtMeshHandle>,
    wv_mat: Res<'w, WeaveMaterialHandle>,
    wv_mesh: Res<'w, WeaveMeshHandle>,
    sa_mat: Res<'w, StringArtMaterialHandle>,
    sa_mesh: Res<'w, StringArtMeshHandle>,
    rd_mat: Res<'w, RdMaterialHandle>,
    rd_mesh: Res<'w, RdMeshHandle>,
    mz_mat: Res<'w, MazeMaterialHandle>,
    mz_mesh: Res<'w, MazeMeshHandle>,
    cb_mat: Res<'w, CircuitMaterialHandle>,
    cb_mesh: Res<'w, CircuitMeshHandle>,
    bp_mat: Res<'w, BubbleMaterialHandle>,
    bp_mesh: Res<'w, BubbleMeshHandle>,
}

#[derive(SystemParam)]
struct ShaderMaterials<'w> {
    ht: ResMut<'w, Assets<HalftoneMaterial>>,
    tp: ResMut<'w, Assets<TopoMaterial>>,
    ms: ResMut<'w, Assets<MosaicMaterial>>,
    eg: ResMut<'w, Assets<EngraveMaterial>>,
    wfc: ResMut<'w, Assets<WfcMaterial>>,
    ck: ResMut<'w, Assets<CrackleMaterial>>,
    ff: ResMut<'w, Assets<FlowFieldMaterial>>,
    oa: ResMut<'w, Assets<OpArtMaterial>>,
    wv: ResMut<'w, Assets<WeaveMaterial>>,
    sa: ResMut<'w, Assets<StringArtMaterial>>,
    rd: ResMut<'w, Assets<RdMaterial>>,
    mz: ResMut<'w, Assets<MazeMaterial>>,
    cb: ResMut<'w, Assets<CircuitMaterial>>,
    bp: ResMut<'w, Assets<BubbleMaterial>>,
}

/// Single query handles visibility for all WGSL-shader quads via StyleId.
#[derive(SystemParam)]
struct StyleVis<'w, 's> {
    q: Query<'w, 's, (&'static mut Visibility, &'static StyleId), Without<TesselMesh>>,
}

/// Bundles the 8 new shader material asset collections for setup().
#[derive(SystemParam)]
struct NewStyleAssets<'w> {
    ff: ResMut<'w, Assets<FlowFieldMaterial>>,
    oa: ResMut<'w, Assets<OpArtMaterial>>,
    wv: ResMut<'w, Assets<WeaveMaterial>>,
    sa: ResMut<'w, Assets<StringArtMaterial>>,
    rd: ResMut<'w, Assets<RdMaterial>>,
    mz: ResMut<'w, Assets<MazeMaterial>>,
    cb: ResMut<'w, Assets<CircuitMaterial>>,
    bp: ResMut<'w, Assets<BubbleMaterial>>,
}

/// Bundles all post-process resources so ui() stays under the 16-param limit.
#[derive(SystemParam)]
struct PostFx<'w, 's> {
    rt: ResMut<'w, SceneRenderTarget>,
    scene_cam: Res<'w, SceneCameraEntity>,
    cam_targets: Query<'w, 's, &'static mut RenderTarget>,
    images: ResMut<'w, Assets<Image>>,
    blur_mat: ResMut<'w, Assets<BlurMaterial>>,
    blur_h: Res<'w, BlurMaterialHandle>,
    blur_mesh: Res<'w, BlurMeshHandle>,
}

// ── App ───────────────────────────────────────────────────────────────────────

fn main() {
    let params = Params::default();
    let (w, h) = (params.out_w, params.out_h);
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Tessellation".into(),
                resolution: bevy::window::WindowResolution::new(w, h),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.03)))
        .insert_resource(params)
        .add_plugins(EguiPlugin::default())
        .add_plugins(Material2dPlugin::<HalftoneMaterial>::default())
        .add_plugins(Material2dPlugin::<TopoMaterial>::default())
        .add_plugins(Material2dPlugin::<MosaicMaterial>::default())
        .add_plugins(Material2dPlugin::<EngraveMaterial>::default())
        .add_plugins(Material2dPlugin::<WfcMaterial>::default())
        .add_plugins(Material2dPlugin::<CrackleMaterial>::default())
        .add_plugins(Material2dPlugin::<FlowFieldMaterial>::default())
        .add_plugins(Material2dPlugin::<OpArtMaterial>::default())
        .add_plugins(Material2dPlugin::<WeaveMaterial>::default())
        .add_plugins(Material2dPlugin::<StringArtMaterial>::default())
        .add_plugins(Material2dPlugin::<RdMaterial>::default())
        .add_plugins(Material2dPlugin::<MazeMaterial>::default())
        .add_plugins(Material2dPlugin::<CircuitMaterial>::default())
        .add_plugins(Material2dPlugin::<BubbleMaterial>::default())
        .add_plugins(Material2dPlugin::<BlurMaterial>::default())
        .add_systems(Startup, setup)
        .add_systems(EguiPrimaryContextPass, ui)
        .run();
}

// ── Components ────────────────────────────────────────────────────────────────

#[derive(Component)]
struct TesselMesh;

// ── Setup ─────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut ht_materials: ResMut<Assets<HalftoneMaterial>>,
    mut topo_materials: ResMut<Assets<TopoMaterial>>,
    mut ms_materials: ResMut<Assets<MosaicMaterial>>,
    mut eg_materials: ResMut<Assets<EngraveMaterial>>,
    mut wfc_materials: ResMut<Assets<WfcMaterial>>,
    mut ck_materials: ResMut<Assets<CrackleMaterial>>,
    mut new_mats: NewStyleAssets,
    mut blur_materials: ResMut<Assets<BlurMaterial>>,
    mut images: ResMut<Assets<Image>>,
    params: Res<Params>,
) {
    // Create intermediate render target; Camera A writes here, blur quad reads it.
    let rt = make_render_target(params.out_w, params.out_h, &mut images);
    commands.insert_resource(SceneRenderTarget(rt.clone()));

    // Camera B: renders the blur quad to the primary window (layer 1).
    // Spawned FIRST so bevy_egui auto-attaches the primary context to this camera
    // (window camera), not to Camera A (render-target camera).  Egui input
    // coordinates are then in screen space, making controls clickable.
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            ..default()
        },
        RenderLayers::layer(1),
    ));

    // Camera A: renders all scene geometry to the render target (layer 0).
    let cam_a = commands
        .spawn((
            Camera2d,
            Camera {
                order: 0,
                ..default()
            },
            RenderTarget::Image(rt.clone().into()),
            RenderLayers::layer(0),
        ))
        .id();
    commands.insert_resource(SceneCameraEntity(cam_a));

    // Low-poly / Voronoi mesh (visible by default)
    commands.spawn((
        Mesh2d(meshes.add(dispatch_mesh(&params))),
        MeshMaterial2d(materials.add(ColorMaterial::from_color(Color::WHITE))),
        TesselMesh,
    ));

    let quad = |w: u32, h: u32| Rectangle::new(w as f32, h as f32);

    // Halftone full-screen quad (hidden by default)
    let ht_mesh = meshes.add(quad(params.out_w, params.out_h));
    let ht_mat = ht_materials.add(HalftoneMaterial {
        uniforms: params.to_ht_uniforms(),
    });
    commands.insert_resource(HalftoneMaterialHandle(ht_mat.clone()));
    commands.insert_resource(HalftoneMeshHandle(ht_mesh.clone()));
    commands.spawn((
        Mesh2d(ht_mesh),
        MeshMaterial2d(ht_mat),
        StyleId(Style::Halftone),
        Visibility::Hidden,
    ));

    // Topo full-screen quad (hidden by default)
    let tp_mesh = meshes.add(quad(params.out_w, params.out_h));
    let tp_mat = topo_materials.add(TopoMaterial {
        uniforms: params.to_topo_uniforms(),
    });
    commands.insert_resource(TopoMaterialHandle(tp_mat.clone()));
    commands.insert_resource(TopoMeshHandle(tp_mesh.clone()));
    commands.spawn((
        Mesh2d(tp_mesh),
        MeshMaterial2d(tp_mat),
        StyleId(Style::Topo),
        Visibility::Hidden,
    ));

    // Mosaic full-screen quad (hidden by default)
    let ms_mesh = meshes.add(quad(params.out_w, params.out_h));
    let ms_mat = ms_materials.add(MosaicMaterial {
        uniforms: params.to_mosaic_uniforms(),
    });
    commands.insert_resource(MosaicMaterialHandle(ms_mat.clone()));
    commands.insert_resource(MosaicMeshHandle(ms_mesh.clone()));
    commands.spawn((
        Mesh2d(ms_mesh),
        MeshMaterial2d(ms_mat),
        StyleId(Style::Mosaic),
        Visibility::Hidden,
    ));

    // Engrave full-screen quad (hidden by default)
    let eg_mesh = meshes.add(quad(params.out_w, params.out_h));
    let eg_mat = eg_materials.add(EngraveMaterial {
        uniforms: params.to_engrave_uniforms(),
    });
    commands.insert_resource(EngraveMaterialHandle(eg_mat.clone()));
    commands.insert_resource(EngraveMeshHandle(eg_mesh.clone()));
    commands.spawn((
        Mesh2d(eg_mesh),
        MeshMaterial2d(eg_mat),
        StyleId(Style::Engrave),
        Visibility::Hidden,
    ));

    // WFC full-screen quad (hidden)
    let wfc_mesh = meshes.add(quad(params.out_w, params.out_h));
    let wfc_mat = wfc_materials.add(WfcMaterial {
        uniforms: params.to_wfc_uniforms(),
    });
    commands.insert_resource(WfcMaterialHandle(wfc_mat.clone()));
    commands.insert_resource(WfcMeshHandle(wfc_mesh.clone()));
    commands.spawn((
        Mesh2d(wfc_mesh),
        MeshMaterial2d(wfc_mat),
        StyleId(Style::Wfc),
        Visibility::Hidden,
    ));

    // Crackle full-screen quad (hidden)
    let ck_mesh = meshes.add(quad(params.out_w, params.out_h));
    let ck_mat = ck_materials.add(CrackleMaterial {
        uniforms: params.to_crackle_uniforms(),
    });
    commands.insert_resource(CrackleMaterialHandle(ck_mat.clone()));
    commands.insert_resource(CrackleMeshHandle(ck_mesh.clone()));
    commands.spawn((
        Mesh2d(ck_mesh),
        MeshMaterial2d(ck_mat),
        StyleId(Style::Crackle),
        Visibility::Hidden,
    ));

    // Flow Field full-screen quad (hidden)
    let ff_mesh = meshes.add(quad(params.out_w, params.out_h));
    let ff_mat = new_mats.ff.add(FlowFieldMaterial {
        uniforms: params.to_ff_uniforms(),
    });
    commands.insert_resource(FlowFieldMaterialHandle(ff_mat.clone()));
    commands.insert_resource(FlowFieldMeshHandle(ff_mesh.clone()));
    commands.spawn((
        Mesh2d(ff_mesh),
        MeshMaterial2d(ff_mat),
        StyleId(Style::FlowField),
        Visibility::Hidden,
    ));

    // Op Art full-screen quad (hidden)
    let oa_mesh = meshes.add(quad(params.out_w, params.out_h));
    let oa_mat = new_mats.oa.add(OpArtMaterial {
        uniforms: params.to_oa_uniforms(),
    });
    commands.insert_resource(OpArtMaterialHandle(oa_mat.clone()));
    commands.insert_resource(OpArtMeshHandle(oa_mesh.clone()));
    commands.spawn((
        Mesh2d(oa_mesh),
        MeshMaterial2d(oa_mat),
        StyleId(Style::OpArt),
        Visibility::Hidden,
    ));

    // Weave full-screen quad (hidden)
    let wv_mesh = meshes.add(quad(params.out_w, params.out_h));
    let wv_mat = new_mats.wv.add(WeaveMaterial {
        uniforms: params.to_wv_uniforms(),
    });
    commands.insert_resource(WeaveMaterialHandle(wv_mat.clone()));
    commands.insert_resource(WeaveMeshHandle(wv_mesh.clone()));
    commands.spawn((
        Mesh2d(wv_mesh),
        MeshMaterial2d(wv_mat),
        StyleId(Style::Weave),
        Visibility::Hidden,
    ));

    // String Art full-screen quad (hidden)
    let sa_mesh = meshes.add(quad(params.out_w, params.out_h));
    let sa_mat = new_mats.sa.add(StringArtMaterial {
        uniforms: params.to_sa_uniforms(),
    });
    commands.insert_resource(StringArtMaterialHandle(sa_mat.clone()));
    commands.insert_resource(StringArtMeshHandle(sa_mesh.clone()));
    commands.spawn((
        Mesh2d(sa_mesh),
        MeshMaterial2d(sa_mat),
        StyleId(Style::StringArt),
        Visibility::Hidden,
    ));

    // Reaction-Diffusion full-screen quad (hidden)
    let rd_mesh = meshes.add(quad(params.out_w, params.out_h));
    let rd_mat = new_mats.rd.add(RdMaterial {
        uniforms: params.to_rd_uniforms(),
    });
    commands.insert_resource(RdMaterialHandle(rd_mat.clone()));
    commands.insert_resource(RdMeshHandle(rd_mesh.clone()));
    commands.spawn((
        Mesh2d(rd_mesh),
        MeshMaterial2d(rd_mat),
        StyleId(Style::RD),
        Visibility::Hidden,
    ));

    // Maze full-screen quad (hidden)
    let mz_mesh = meshes.add(quad(params.out_w, params.out_h));
    let mz_mat = new_mats.mz.add(MazeMaterial {
        uniforms: params.to_mz_uniforms(),
    });
    commands.insert_resource(MazeMaterialHandle(mz_mat.clone()));
    commands.insert_resource(MazeMeshHandle(mz_mesh.clone()));
    commands.spawn((
        Mesh2d(mz_mesh),
        MeshMaterial2d(mz_mat),
        StyleId(Style::Maze),
        Visibility::Hidden,
    ));

    // Circuit Board full-screen quad (hidden)
    let cb_mesh = meshes.add(quad(params.out_w, params.out_h));
    let cb_mat = new_mats.cb.add(CircuitMaterial {
        uniforms: params.to_cb_uniforms(),
    });
    commands.insert_resource(CircuitMaterialHandle(cb_mat.clone()));
    commands.insert_resource(CircuitMeshHandle(cb_mesh.clone()));
    commands.spawn((
        Mesh2d(cb_mesh),
        MeshMaterial2d(cb_mat),
        StyleId(Style::Circuit),
        Visibility::Hidden,
    ));

    // Bubble Pack full-screen quad (hidden)
    let bp_mesh = meshes.add(quad(params.out_w, params.out_h));
    let bp_mat = new_mats.bp.add(BubbleMaterial {
        uniforms: params.to_bp_uniforms(),
    });
    commands.insert_resource(BubbleMaterialHandle(bp_mat.clone()));
    commands.insert_resource(BubbleMeshHandle(bp_mesh.clone()));
    commands.spawn((
        Mesh2d(bp_mesh),
        MeshMaterial2d(bp_mat),
        StyleId(Style::Bubble),
        Visibility::Hidden,
    ));

    // Blur quad: always visible on layer 1, sampled by Camera B.
    let (bw, bh) = (params.out_w as f32, params.out_h as f32);
    let blur_mesh = meshes.add(Rectangle::new(bw, bh));
    let blur_mat = blur_materials.add(BlurMaterial {
        uniforms: BlurUniforms {
            radius: params.blur_radius,
            inv_w: 1.0 / bw,
            inv_h: 1.0 / bh,
            _pad: 0.0,
        },
        source: rt,
    });
    commands.insert_resource(BlurMaterialHandle(blur_mat.clone()));
    commands.insert_resource(BlurMeshHandle(blur_mesh.clone()));
    commands.spawn((
        Mesh2d(blur_mesh),
        MeshMaterial2d(blur_mat),
        RenderLayers::layer(1),
    ));
}

// ── UI ────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn ui(
    mut ctx: EguiContexts,
    mut params: ResMut<Params>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ShaderMaterials,
    mut commands: Commands,
    mut windows: Query<&mut Window, With<bevy::window::PrimaryWindow>>,
    handles: ShaderHandles,
    mut lp_query: Query<(&Mesh2d, &mut Visibility), With<TesselMesh>>,
    mut sv: StyleVis,
    mut post: PostFx,
) -> Result {
    let mut dirty = false;
    let mut ht_dirty = false;
    let mut tp_dirty = false;
    let mut ms_dirty = false;
    let mut eg_dirty = false;
    let mut wfc_grid_dirty = false;
    let mut wfc_vis_dirty = false;
    let mut ck_dirty = false;
    let mut ff_dirty = false;
    let mut oa_dirty = false;
    let mut wv_dirty = false;
    let mut sa_dirty = false;
    let mut rd_dirty = false;
    let mut mz_grid_dirty = false;
    let mut mz_vis_dirty = false;
    let mut cb_dirty = false;
    let mut bp_dirty = false;
    let mut blur_dirty = false;
    let mut resize = false;
    let prev_tab = params.tab;

    // If a screenshot was requested last frame, take it now (UI not rendered this frame)
    if params.screenshot_pending {
        params.screenshot_pending = false;
        let path = format!(
            "tessellation_{}.png",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        info!("Saving {path}");
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
        return Ok(());
    }

    let ctx = ctx.ctx_mut()?;

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show(ctx, |_ui| {});

    egui::Window::new("Parameters")
        .resizable(false)
        .constrain(true)
        .default_width(480.0)
        .show(ctx, |ui| {
            // ── Tabs ─────────────────────────────────────────────────────────
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut params.tab, Tab::LowPoly, "Low Poly");
                ui.selectable_value(&mut params.tab, Tab::Voronoi, "Voronoi");
                ui.selectable_value(&mut params.tab, Tab::Halftone, "Halftone");
                ui.selectable_value(&mut params.tab, Tab::Topo, "Topo");
                ui.selectable_value(&mut params.tab, Tab::Mosaic, "Mosaic");
                ui.selectable_value(&mut params.tab, Tab::Engrave, "Engrave");
                ui.selectable_value(&mut params.tab, Tab::Wfc, "WFC");
                ui.selectable_value(&mut params.tab, Tab::Crackle, "Crackle");
                ui.selectable_value(&mut params.tab, Tab::FlowField, "Flow Field");
                ui.selectable_value(&mut params.tab, Tab::OpArt, "Op Art");
                ui.selectable_value(&mut params.tab, Tab::Weave, "Weave");
                ui.selectable_value(&mut params.tab, Tab::StringArt, "String Art");
                ui.selectable_value(&mut params.tab, Tab::RD, "React-Diff");
                ui.selectable_value(&mut params.tab, Tab::Maze, "Maze");
                ui.selectable_value(&mut params.tab, Tab::Circuit, "Circuit");
                ui.selectable_value(&mut params.tab, Tab::Bubble, "Bubble");
                ui.selectable_value(&mut params.tab, Tab::Penrose, "Penrose");
                ui.selectable_value(&mut params.tab, Tab::Output, "Output");
            });
            ui.separator();

            match params.tab {
                // ── Low Poly ──────────────────────────────────────────────────
                Tab::LowPoly => {
                    egui::Grid::new("lp")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            lp_slider(ui, "Columns", &mut params.lp_cols, 4..=60, &mut dirty);
                            lp_slider(ui, "Rows", &mut params.lp_rows, 2..=36, &mut dirty);
                            fp_slider(ui, "Jitter", &mut params.lp_jitter, 0.0..=1.0, &mut dirty);
                            fp_slider(
                                ui,
                                "Color variation",
                                &mut params.lp_color_var,
                                0.0..=0.4,
                                &mut dirty,
                            );
                            color_row(ui, "Top-left", &mut params.lp_c_tl, &mut dirty);
                            color_row(ui, "Top-right", &mut params.lp_c_tr, &mut dirty);
                            color_row(ui, "Bottom-left", &mut params.lp_c_bl, &mut dirty);
                            color_row(ui, "Bottom-right", &mut params.lp_c_br, &mut dirty);
                        });
                }

                // ── Voronoi ───────────────────────────────────────────────────
                Tab::Voronoi => {
                    egui::Grid::new("vor")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            lp_slider(ui, "Columns", &mut params.vor_cols, 4..=40, &mut dirty);
                            lp_slider(ui, "Rows", &mut params.vor_rows, 2..=24, &mut dirty);
                            fp_slider(ui, "Jitter", &mut params.vor_jitter, 0.0..=1.0, &mut dirty);
                            fp_slider(ui, "Border", &mut params.vor_border, 0.0..=30.0, &mut dirty);
                            fp_slider(
                                ui,
                                "Dome height",
                                &mut params.vor_elevation,
                                0.0..=0.3,
                                &mut dirty,
                            );
                            fp_slider(
                                ui,
                                "Color variation",
                                &mut params.vor_color_var,
                                0.0..=0.2,
                                &mut dirty,
                            );
                            color_row(ui, "Top-left", &mut params.vor_c_tl, &mut dirty);
                            color_row(ui, "Top-right", &mut params.vor_c_tr, &mut dirty);
                            color_row(ui, "Bottom-left", &mut params.vor_c_bl, &mut dirty);
                            color_row(ui, "Bottom-right", &mut params.vor_c_br, &mut dirty);
                        });
                }

                // ── Halftone ──────────────────────────────────────────────────
                Tab::Halftone => {
                    egui::Grid::new("ht")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            fp_slider(
                                ui,
                                "Tile frequency",
                                &mut params.ht_tile_freq,
                                1.0..=12.0,
                                &mut ht_dirty,
                            );
                            fp_slider(
                                ui,
                                "Dot density",
                                &mut params.ht_dot_density,
                                10.0..=120.0,
                                &mut ht_dirty,
                            );
                            fp_slider(
                                ui,
                                "Exponent",
                                &mut params.ht_exponent,
                                0.5..=5.0,
                                &mut ht_dirty,
                            );
                            fp_slider(
                                ui,
                                "Ambient",
                                &mut params.ht_ambient,
                                0.0..=0.5,
                                &mut ht_dirty,
                            );
                            fp_slider(
                                ui,
                                "Diffuse",
                                &mut params.ht_diffuse,
                                0.0..=1.0,
                                &mut ht_dirty,
                            );
                            fp_slider(
                                ui,
                                "Specular",
                                &mut params.ht_specular,
                                0.0..=1.0,
                                &mut ht_dirty,
                            );
                            fp_slider(
                                ui,
                                "Shininess",
                                &mut params.ht_shininess,
                                1.0..=128.0,
                                &mut ht_dirty,
                            );
                            fp_slider_suffix(
                                ui,
                                "Light elevation",
                                &mut params.ht_light_elev,
                                0.0..=90.0,
                                "°",
                                &mut ht_dirty,
                            );
                            fp_slider_suffix(
                                ui,
                                "Light azimuth",
                                &mut params.ht_light_az,
                                0.0..=360.0,
                                "°",
                                &mut ht_dirty,
                            );
                            fp_slider(
                                ui,
                                "Color variation",
                                &mut params.ht_color_var,
                                0.0..=0.2,
                                &mut ht_dirty,
                            );
                            color_row(ui, "Top-left", &mut params.ht_c_tl, &mut ht_dirty);
                            color_row(ui, "Top-right", &mut params.ht_c_tr, &mut ht_dirty);
                            color_row(ui, "Bottom-left", &mut params.ht_c_bl, &mut ht_dirty);
                            color_row(ui, "Bottom-right", &mut params.ht_c_br, &mut ht_dirty);
                        });
                }

                // ── Topo ──────────────────────────────────────────────────────
                Tab::Topo => {
                    egui::Grid::new("tp")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            fp_slider(
                                ui,
                                "Noise scale",
                                &mut params.tp_noise_scale,
                                0.5..=10.0,
                                &mut tp_dirty,
                            );
                            fp_slider(
                                ui,
                                "Octaves",
                                &mut params.tp_octaves,
                                1.0..=8.0,
                                &mut tp_dirty,
                            );
                            fp_slider(
                                ui,
                                "Line count",
                                &mut params.tp_line_count,
                                4.0..=60.0,
                                &mut tp_dirty,
                            );
                            fp_slider(
                                ui,
                                "Line width",
                                &mut params.tp_line_width,
                                0.005..=0.2,
                                &mut tp_dirty,
                            );
                            fp_slider(
                                ui,
                                "Ambient",
                                &mut params.tp_ambient,
                                0.0..=0.5,
                                &mut tp_dirty,
                            );
                            fp_slider(
                                ui,
                                "Diffuse",
                                &mut params.tp_diffuse,
                                0.0..=1.0,
                                &mut tp_dirty,
                            );
                            fp_slider(
                                ui,
                                "Specular",
                                &mut params.tp_specular,
                                0.0..=1.0,
                                &mut tp_dirty,
                            );
                            fp_slider(
                                ui,
                                "Shininess",
                                &mut params.tp_shininess,
                                1.0..=128.0,
                                &mut tp_dirty,
                            );
                            fp_slider_suffix(
                                ui,
                                "Light elevation",
                                &mut params.tp_light_elev,
                                0.0..=90.0,
                                "°",
                                &mut tp_dirty,
                            );
                            fp_slider_suffix(
                                ui,
                                "Light azimuth",
                                &mut params.tp_light_az,
                                0.0..=360.0,
                                "°",
                                &mut tp_dirty,
                            );
                            color_row(ui, "Background", &mut params.tp_bg_color, &mut tp_dirty);
                            color_row(ui, "Line color", &mut params.tp_line_color, &mut tp_dirty);
                        });
                }

                // ── Mosaic ────────────────────────────────────────────────────
                Tab::Mosaic => {
                    egui::Grid::new("ms")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            fp_slider(
                                ui,
                                "Tile count",
                                &mut params.ms_tile_count,
                                4.0..=80.0,
                                &mut ms_dirty,
                            );
                            fp_slider(
                                ui,
                                "Grout width",
                                &mut params.ms_grout_width,
                                0.01..=0.20,
                                &mut ms_dirty,
                            );
                            fp_slider(
                                ui,
                                "Bevel width",
                                &mut params.ms_bevel_width,
                                0.0..=0.5,
                                &mut ms_dirty,
                            );
                            fp_slider(
                                ui,
                                "Noise scale",
                                &mut params.ms_noise_scale,
                                0.5..=8.0,
                                &mut ms_dirty,
                            );
                            fp_slider(
                                ui,
                                "Color variation",
                                &mut params.ms_color_var,
                                0.0..=0.3,
                                &mut ms_dirty,
                            );
                            fp_slider(
                                ui,
                                "Ambient",
                                &mut params.ms_ambient,
                                0.0..=0.5,
                                &mut ms_dirty,
                            );
                            fp_slider(
                                ui,
                                "Diffuse",
                                &mut params.ms_diffuse,
                                0.0..=1.0,
                                &mut ms_dirty,
                            );
                            fp_slider(
                                ui,
                                "Specular",
                                &mut params.ms_specular,
                                0.0..=1.0,
                                &mut ms_dirty,
                            );
                            fp_slider(
                                ui,
                                "Shininess",
                                &mut params.ms_shininess,
                                1.0..=128.0,
                                &mut ms_dirty,
                            );
                            fp_slider_suffix(
                                ui,
                                "Light elevation",
                                &mut params.ms_light_elev,
                                0.0..=90.0,
                                "°",
                                &mut ms_dirty,
                            );
                            fp_slider_suffix(
                                ui,
                                "Light azimuth",
                                &mut params.ms_light_az,
                                0.0..=360.0,
                                "°",
                                &mut ms_dirty,
                            );
                            color_row(ui, "Grout", &mut params.ms_grout_color, &mut ms_dirty);
                            color_row(ui, "Top-left", &mut params.ms_c_tl, &mut ms_dirty);
                            color_row(ui, "Top-right", &mut params.ms_c_tr, &mut ms_dirty);
                            color_row(ui, "Bottom-left", &mut params.ms_c_bl, &mut ms_dirty);
                            color_row(ui, "Bottom-right", &mut params.ms_c_br, &mut ms_dirty);
                        });
                }

                // ── Engrave ───────────────────────────────────────────────────
                Tab::Engrave => {
                    egui::Grid::new("eg")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            fp_slider(
                                ui,
                                "Noise scale",
                                &mut params.eg_noise_scale,
                                0.5..=8.0,
                                &mut eg_dirty,
                            );
                            fp_slider(
                                ui,
                                "Octaves",
                                &mut params.eg_octaves,
                                1.0..=8.0,
                                &mut eg_dirty,
                            );
                            fp_slider(
                                ui,
                                "Line count",
                                &mut params.eg_line_count,
                                4.0..=60.0,
                                &mut eg_dirty,
                            );
                            fp_slider(
                                ui,
                                "Dot density",
                                &mut params.eg_dot_density,
                                20.0..=300.0,
                                &mut eg_dirty,
                            );
                            fp_slider(
                                ui,
                                "Min dot radius",
                                &mut params.eg_dot_min_r,
                                0.0..=0.4,
                                &mut eg_dirty,
                            );
                            fp_slider(
                                ui,
                                "Max dot radius",
                                &mut params.eg_dot_max_r,
                                0.1..=0.5,
                                &mut eg_dirty,
                            );
                            color_row(ui, "Background", &mut params.eg_bg_color, &mut eg_dirty);
                            color_row(ui, "Dot color", &mut params.eg_dot_color, &mut eg_dirty);
                        });
                }

                // ── WFC ───────────────────────────────────────────────────────
                Tab::Wfc => {
                    egui::Grid::new("wfc")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            lp_slider(
                                ui,
                                "Grid width",
                                &mut params.wfc_grid_w,
                                4..=32,
                                &mut wfc_grid_dirty,
                            );
                            lp_slider(
                                ui,
                                "Grid height",
                                &mut params.wfc_grid_h,
                                4..=24,
                                &mut wfc_grid_dirty,
                            );
                            fp_slider(
                                ui,
                                "Line width",
                                &mut params.wfc_line_width,
                                0.02..=0.45,
                                &mut wfc_vis_dirty,
                            );
                            fp_slider(
                                ui,
                                "Color variation",
                                &mut params.wfc_color_var,
                                0.0..=0.3,
                                &mut wfc_vis_dirty,
                            );

                            ui.label("Tile set");
                            let changed = ui
                                .radio_value(&mut params.wfc_tile_set, 0, "Pipes")
                                .changed()
                                | ui.radio_value(&mut params.wfc_tile_set, 1, "+Cross")
                                    .changed()
                                | ui.radio_value(&mut params.wfc_tile_set, 2, "+T-junc")
                                    .changed();
                            if changed {
                                wfc_grid_dirty = true;
                            }
                            ui.end_row();

                            color_row(
                                ui,
                                "Pipe color",
                                &mut params.wfc_fg_color,
                                &mut wfc_vis_dirty,
                            );
                            color_row(
                                ui,
                                "Background",
                                &mut params.wfc_bg_color,
                                &mut wfc_vis_dirty,
                            );
                        });
                }

                // ── Crackle ───────────────────────────────────────────────────
                Tab::Crackle => {
                    egui::Grid::new("ck")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            fp_slider(
                                ui,
                                "Cell count",
                                &mut params.ck_cell_count,
                                4.0..=40.0,
                                &mut ck_dirty,
                            );
                            fp_slider(
                                ui,
                                "Crack width",
                                &mut params.ck_crack_width,
                                0.01..=0.3,
                                &mut ck_dirty,
                            );
                            fp_slider(
                                ui,
                                "Jitter",
                                &mut params.ck_jitter,
                                0.0..=1.0,
                                &mut ck_dirty,
                            );
                            fp_slider(
                                ui,
                                "Domain warp",
                                &mut params.ck_warp,
                                0.0..=1.5,
                                &mut ck_dirty,
                            );
                            fp_slider(
                                ui,
                                "Color variation",
                                &mut params.ck_color_var,
                                0.0..=0.3,
                                &mut ck_dirty,
                            );
                            fp_slider(
                                ui,
                                "Inner vein width",
                                &mut params.ck_inner_width,
                                0.0..=0.5,
                                &mut ck_dirty,
                            );
                            fp_slider(
                                ui,
                                "Inner vein bright",
                                &mut params.ck_inner_bright,
                                0.0..=1.0,
                                &mut ck_dirty,
                            );
                            fp_slider(
                                ui,
                                "Cell gradient",
                                &mut params.ck_cell_depth,
                                0.0..=1.0,
                                &mut ck_dirty,
                            );
                            color_row(ui, "Cell color", &mut params.ck_bg_color, &mut ck_dirty);
                            color_row(ui, "Crack color", &mut params.ck_crack_color, &mut ck_dirty);
                            color_row(ui, "Top-left", &mut params.ck_c_tl, &mut ck_dirty);
                            color_row(ui, "Top-right", &mut params.ck_c_tr, &mut ck_dirty);
                            color_row(ui, "Bottom-left", &mut params.ck_c_bl, &mut ck_dirty);
                            color_row(ui, "Bottom-right", &mut params.ck_c_br, &mut ck_dirty);
                        });
                }

                // ── Flow Field ────────────────────────────────────────────────
                Tab::FlowField => {
                    egui::Grid::new("ff")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            fp_slider(
                                ui,
                                "Frequency",
                                &mut params.ff_freq,
                                0.5..=10.0,
                                &mut ff_dirty,
                            );
                            fp_slider(
                                ui,
                                "Line count",
                                &mut params.ff_n_lines,
                                2.0..=40.0,
                                &mut ff_dirty,
                            );
                            fp_slider(
                                ui,
                                "Line width",
                                &mut params.ff_line_width,
                                0.01..=0.5,
                                &mut ff_dirty,
                            );
                            fp_slider(
                                ui,
                                "Domain warp",
                                &mut params.ff_warp,
                                0.0..=2.0,
                                &mut ff_dirty,
                            );
                            fp_slider(
                                ui,
                                "Octaves",
                                &mut params.ff_octaves,
                                1.0..=8.0,
                                &mut ff_dirty,
                            );
                            fp_slider(
                                ui,
                                "Color variation",
                                &mut params.ff_color_var,
                                0.0..=0.3,
                                &mut ff_dirty,
                            );
                            color_row(ui, "Top-left", &mut params.ff_c_tl, &mut ff_dirty);
                            color_row(ui, "Top-right", &mut params.ff_c_tr, &mut ff_dirty);
                            color_row(ui, "Bottom-left", &mut params.ff_c_bl, &mut ff_dirty);
                            color_row(ui, "Bottom-right", &mut params.ff_c_br, &mut ff_dirty);
                        });
                }

                // ── Op Art ────────────────────────────────────────────────────
                Tab::OpArt => {
                    egui::Grid::new("oa")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            fp_slider(
                                ui,
                                "Ring count",
                                &mut params.oa_ring_count,
                                2.0..=24.0,
                                &mut oa_dirty,
                            );
                            fp_slider(
                                ui,
                                "Ring width",
                                &mut params.oa_ring_width,
                                0.02..=1.0,
                                &mut oa_dirty,
                            );
                            fp_slider(ui, "Warp", &mut params.oa_warp, 0.0..=0.6, &mut oa_dirty);
                            fp_slider(
                                ui,
                                "Warp freq",
                                &mut params.oa_freq,
                                0.5..=8.0,
                                &mut oa_dirty,
                            );
                            fp_slider(ui, "Twist", &mut params.oa_twist, 0.0..=4.0, &mut oa_dirty);
                            color_row(ui, "Foreground", &mut params.oa_fg_color, &mut oa_dirty);
                            color_row(ui, "Background", &mut params.oa_bg_color, &mut oa_dirty);
                        });
                }

                // ── Weave ─────────────────────────────────────────────────────
                Tab::Weave => {
                    egui::Grid::new("wv")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            fp_slider(
                                ui,
                                "Thread count",
                                &mut params.wv_thread_count,
                                4.0..=50.0,
                                &mut wv_dirty,
                            );
                            fp_slider(
                                ui,
                                "Thread width",
                                &mut params.wv_thread_width,
                                0.1..=0.9,
                                &mut wv_dirty,
                            );
                            fp_slider(
                                ui,
                                "Shadow",
                                &mut params.wv_shadow,
                                0.0..=0.8,
                                &mut wv_dirty,
                            );
                            fp_slider(
                                ui,
                                "Color var",
                                &mut params.wv_color_var,
                                0.0..=0.3,
                                &mut wv_dirty,
                            );
                            color_row(
                                ui,
                                "Warp (vertical)",
                                &mut params.wv_warp_color,
                                &mut wv_dirty,
                            );
                            color_row(
                                ui,
                                "Weft (horizontal)",
                                &mut params.wv_weft_color,
                                &mut wv_dirty,
                            );
                            color_row(ui, "Gap/background", &mut params.wv_bg_color, &mut wv_dirty);
                        });
                }

                // ── String Art ────────────────────────────────────────────────
                Tab::StringArt => {
                    egui::Grid::new("sa")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            fp_slider(
                                ui,
                                "Points",
                                &mut params.sa_n_points,
                                20.0..=256.0,
                                &mut sa_dirty,
                            );
                            fp_slider(
                                ui,
                                "Offset k",
                                &mut params.sa_k_offset,
                                2.0..=200.0,
                                &mut sa_dirty,
                            );
                            fp_slider(
                                ui,
                                "Line width",
                                &mut params.sa_line_width,
                                0.001..=0.012,
                                &mut sa_dirty,
                            );
                            ui.label("Circles");
                            let changed = ui
                                .radio_value(&mut params.sa_n_circles, 1.0, "One")
                                .changed()
                                | ui.radio_value(&mut params.sa_n_circles, 2.0, "Two")
                                    .changed();
                            if changed {
                                sa_dirty = true;
                            }
                            ui.end_row();
                            fp_slider(
                                ui,
                                "Circle A x",
                                &mut params.sa_cx_a,
                                0.1..=0.9,
                                &mut sa_dirty,
                            );
                            fp_slider(
                                ui,
                                "Circle A y",
                                &mut params.sa_cy_a,
                                0.1..=0.9,
                                &mut sa_dirty,
                            );
                            fp_slider(
                                ui,
                                "Circle A r",
                                &mut params.sa_r_a,
                                0.05..=0.48,
                                &mut sa_dirty,
                            );
                            fp_slider(
                                ui,
                                "Circle B x",
                                &mut params.sa_cx_b,
                                0.1..=0.9,
                                &mut sa_dirty,
                            );
                            fp_slider(
                                ui,
                                "Circle B y",
                                &mut params.sa_cy_b,
                                0.1..=0.9,
                                &mut sa_dirty,
                            );
                            fp_slider(
                                ui,
                                "Circle B r",
                                &mut params.sa_r_b,
                                0.05..=0.48,
                                &mut sa_dirty,
                            );
                            color_row(ui, "String color", &mut params.sa_fg_color, &mut sa_dirty);
                            color_row(ui, "Background", &mut params.sa_bg_color, &mut sa_dirty);
                        });
                }

                // ── Reaction-Diffusion ────────────────────────────────────────
                Tab::RD => {
                    egui::Grid::new("rd")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            fp_slider(ui, "Scale", &mut params.rd_scale, 1.0..=10.0, &mut rd_dirty);
                            fp_slider(
                                ui,
                                "Sharpness",
                                &mut params.rd_sharpness,
                                1.0..=20.0,
                                &mut rd_dirty,
                            );
                            fp_slider(ui, "Warp", &mut params.rd_warp, 0.0..=1.5, &mut rd_dirty);
                            fp_slider(
                                ui,
                                "Balance",
                                &mut params.rd_balance,
                                -0.5..=0.5,
                                &mut rd_dirty,
                            );
                            color_row(ui, "Species A", &mut params.rd_fg_color, &mut rd_dirty);
                            color_row(ui, "Background", &mut params.rd_bg_color, &mut rd_dirty);
                            color_row(ui, "Top-left", &mut params.rd_c_tl, &mut rd_dirty);
                            color_row(ui, "Top-right", &mut params.rd_c_tr, &mut rd_dirty);
                            color_row(ui, "Bottom-left", &mut params.rd_c_bl, &mut rd_dirty);
                            color_row(ui, "Bottom-right", &mut params.rd_c_br, &mut rd_dirty);
                        });
                }

                // ── Maze ──────────────────────────────────────────────────────
                Tab::Maze => {
                    egui::Grid::new("mz")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            lp_slider(
                                ui,
                                "Grid width",
                                &mut params.mz_grid_w,
                                4..=32,
                                &mut mz_grid_dirty,
                            );
                            lp_slider(
                                ui,
                                "Grid height",
                                &mut params.mz_grid_h,
                                4..=24,
                                &mut mz_grid_dirty,
                            );
                            fp_slider(
                                ui,
                                "Wall width",
                                &mut params.mz_line_width,
                                0.01..=0.45,
                                &mut mz_vis_dirty,
                            );
                            color_row(
                                ui,
                                "Wall color",
                                &mut params.mz_wall_color,
                                &mut mz_vis_dirty,
                            );
                            color_row(
                                ui,
                                "Floor color",
                                &mut params.mz_floor_color,
                                &mut mz_vis_dirty,
                            );
                        });
                }

                // ── Circuit Board ─────────────────────────────────────────────
                Tab::Circuit => {
                    egui::Grid::new("cb")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            fp_slider(
                                ui,
                                "Grid density",
                                &mut params.cb_cell_size,
                                4.0..=30.0,
                                &mut cb_dirty,
                            );
                            fp_slider(
                                ui,
                                "Trace width",
                                &mut params.cb_trace_width,
                                0.02..=0.4,
                                &mut cb_dirty,
                            );
                            fp_slider(
                                ui,
                                "Via radius",
                                &mut params.cb_via_radius,
                                0.05..=0.3,
                                &mut cb_dirty,
                            );
                            fp_slider(
                                ui,
                                "Density",
                                &mut params.cb_density,
                                0.1..=1.0,
                                &mut cb_dirty,
                            );
                            fp_slider(
                                ui,
                                "Color var",
                                &mut params.cb_color_var,
                                0.0..=0.3,
                                &mut cb_dirty,
                            );
                            color_row(ui, "Trace", &mut params.cb_trace_color, &mut cb_dirty);
                            color_row(ui, "Via", &mut params.cb_via_color, &mut cb_dirty);
                            color_row(ui, "Background", &mut params.cb_bg_color, &mut cb_dirty);
                        });
                }

                // ── Bubble Pack ───────────────────────────────────────────────
                Tab::Bubble => {
                    egui::Grid::new("bp")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            fp_slider(
                                ui,
                                "Density",
                                &mut params.bp_density,
                                3.0..=20.0,
                                &mut bp_dirty,
                            );
                            fp_slider(
                                ui,
                                "Min radius",
                                &mut params.bp_min_r,
                                0.05..=0.4,
                                &mut bp_dirty,
                            );
                            fp_slider(
                                ui,
                                "Max radius",
                                &mut params.bp_max_r,
                                0.15..=0.48,
                                &mut bp_dirty,
                            );
                            fp_slider(
                                ui,
                                "Border",
                                &mut params.bp_border,
                                0.0..=0.15,
                                &mut bp_dirty,
                            );
                            fp_slider(
                                ui,
                                "Jitter",
                                &mut params.bp_jitter,
                                0.0..=1.0,
                                &mut bp_dirty,
                            );
                            fp_slider(
                                ui,
                                "Color var",
                                &mut params.bp_color_var,
                                0.0..=0.3,
                                &mut bp_dirty,
                            );
                            color_row(ui, "Top-left", &mut params.bp_c_tl, &mut bp_dirty);
                            color_row(ui, "Top-right", &mut params.bp_c_tr, &mut bp_dirty);
                            color_row(ui, "Bottom-left", &mut params.bp_c_bl, &mut bp_dirty);
                            color_row(ui, "Bottom-right", &mut params.bp_c_br, &mut bp_dirty);
                        });
                }

                // ── Penrose ───────────────────────────────────────────────────
                Tab::Penrose => {
                    egui::Grid::new("pe")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            lp_slider(
                                ui,
                                "Iterations",
                                &mut params.pe_iterations,
                                2..=7,
                                &mut dirty,
                            );
                            fp_slider(
                                ui,
                                "Color var",
                                &mut params.pe_color_var,
                                0.0..=0.3,
                                &mut dirty,
                            );
                            color_row(ui, "Kite color", &mut params.pe_color_a, &mut dirty);
                            color_row(ui, "Dart color", &mut params.pe_color_b, &mut dirty);
                        });
                }

                // ── Output ────────────────────────────────────────────────────
                Tab::Output => {
                    egui::Grid::new("out")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Width");
                            if ui
                                .add(egui::Slider::new(&mut params.out_w, 320..=3840))
                                .changed()
                            {
                                dirty = true;
                                resize = true;
                            }
                            ui.end_row();
                            ui.label("Height");
                            if ui
                                .add(egui::Slider::new(&mut params.out_h, 240..=2160))
                                .changed()
                            {
                                dirty = true;
                                resize = true;
                            }
                            ui.end_row();
                            ui.label("Blur radius");
                            if ui
                                .add(
                                    egui::Slider::new(&mut params.blur_radius, 0.0..=20.0)
                                        .suffix("px"),
                                )
                                .changed()
                            {
                                blur_dirty = true;
                            }
                            ui.end_row();
                        });

                    ui.label(
                        egui::RichText::new(format!("{}  ×  {}", params.out_w, params.out_h))
                            .weak(),
                    );
                    ui.add_space(6.0);

                    if matches!(params.style, Style::LowPoly | Style::Voronoi) {
                        ui.label("Gradient corners");
                        ui.add_space(2.0);
                        egui::Grid::new("out_col")
                            .num_columns(2)
                            .spacing([8.0, 4.0])
                            .show(ui, |ui| match params.style {
                                Style::LowPoly => {
                                    color_row(ui, "Top-left", &mut params.lp_c_tl, &mut dirty);
                                    color_row(ui, "Top-right", &mut params.lp_c_tr, &mut dirty);
                                    color_row(ui, "Bottom-left", &mut params.lp_c_bl, &mut dirty);
                                    color_row(ui, "Bottom-right", &mut params.lp_c_br, &mut dirty);
                                }
                                Style::Voronoi => {
                                    color_row(ui, "Top-left", &mut params.vor_c_tl, &mut dirty);
                                    color_row(ui, "Top-right", &mut params.vor_c_tr, &mut dirty);
                                    color_row(ui, "Bottom-left", &mut params.vor_c_bl, &mut dirty);
                                    color_row(ui, "Bottom-right", &mut params.vor_c_br, &mut dirty);
                                }
                                _ => {}
                            });
                    }
                }
            }

            ui.separator();
            ui.horizontal(|ui| {
                if matches!(
                    params.style,
                    Style::LowPoly | Style::Voronoi | Style::Wfc | Style::Maze | Style::Penrose
                ) && ui.button("Randomize").clicked()
                {
                    params.seed = rand::random();
                    match params.style {
                        Style::Wfc => wfc_grid_dirty = true,
                        Style::Maze => mz_grid_dirty = true,
                        _ => dirty = true,
                    }
                }
                if ui.button("Save PNG").clicked() {
                    params.screenshot_pending = true;
                }
            });
        });

    // Tab switch → update style and mark dirty
    if params.tab != prev_tab {
        match params.tab {
            Tab::LowPoly => {
                params.style = Style::LowPoly;
                dirty = true;
            }
            Tab::Voronoi => {
                params.style = Style::Voronoi;
                dirty = true;
            }
            Tab::Halftone => {
                params.style = Style::Halftone;
                ht_dirty = true;
            }
            Tab::Topo => {
                params.style = Style::Topo;
                tp_dirty = true;
            }
            Tab::Mosaic => {
                params.style = Style::Mosaic;
                ms_dirty = true;
            }
            Tab::Engrave => {
                params.style = Style::Engrave;
                eg_dirty = true;
            }
            Tab::Wfc => {
                params.style = Style::Wfc;
                wfc_vis_dirty = true;
            }
            Tab::Crackle => {
                params.style = Style::Crackle;
                ck_dirty = true;
            }
            Tab::FlowField => {
                params.style = Style::FlowField;
                ff_dirty = true;
            }
            Tab::OpArt => {
                params.style = Style::OpArt;
                oa_dirty = true;
            }
            Tab::Weave => {
                params.style = Style::Weave;
                wv_dirty = true;
            }
            Tab::StringArt => {
                params.style = Style::StringArt;
                sa_dirty = true;
            }
            Tab::RD => {
                params.style = Style::RD;
                rd_dirty = true;
            }
            Tab::Maze => {
                params.style = Style::Maze;
                mz_vis_dirty = true;
            }
            Tab::Circuit => {
                params.style = Style::Circuit;
                cb_dirty = true;
            }
            Tab::Bubble => {
                params.style = Style::Bubble;
                bp_dirty = true;
            }
            Tab::Penrose => {
                params.style = Style::Penrose;
                dirty = true;
            }
            Tab::Output => {}
        }
    }

    if resize {
        if let Ok(mut window) = windows.single_mut() {
            window.resolution = bevy::window::WindowResolution::new(params.out_w, params.out_h);
        }
        let quad: Mesh = Rectangle::new(params.out_w as f32, params.out_h as f32).into();
        if let Some(mesh) = meshes.get_mut(&handles.ht_mesh.0) {
            *mesh = quad.clone();
        }
        if let Some(mesh) = meshes.get_mut(&handles.tp_mesh.0) {
            *mesh = quad.clone();
        }
        if let Some(mesh) = meshes.get_mut(&handles.ms_mesh.0) {
            *mesh = quad.clone();
        }
        if let Some(mesh) = meshes.get_mut(&handles.eg_mesh.0) {
            *mesh = quad.clone();
        }
        if let Some(mesh) = meshes.get_mut(&handles.wfc_mesh.0) {
            *mesh = quad.clone();
        }
        if let Some(mesh) = meshes.get_mut(&handles.ck_mesh.0) {
            *mesh = quad.clone();
        }
        if let Some(mesh) = meshes.get_mut(&handles.ff_mesh.0) {
            *mesh = quad.clone();
        }
        if let Some(mesh) = meshes.get_mut(&handles.oa_mesh.0) {
            *mesh = quad.clone();
        }
        if let Some(mesh) = meshes.get_mut(&handles.wv_mesh.0) {
            *mesh = quad.clone();
        }
        if let Some(mesh) = meshes.get_mut(&handles.sa_mesh.0) {
            *mesh = quad.clone();
        }
        if let Some(mesh) = meshes.get_mut(&handles.rd_mesh.0) {
            *mesh = quad.clone();
        }
        if let Some(mesh) = meshes.get_mut(&handles.mz_mesh.0) {
            *mesh = quad.clone();
        }
        if let Some(mesh) = meshes.get_mut(&handles.cb_mesh.0) {
            *mesh = quad.clone();
        }
        if let Some(mesh) = meshes.get_mut(&handles.bp_mesh.0) {
            *mesh = quad.clone();
        }
        // Recreate render target at new size, update Camera A target and blur material.
        let new_rt = make_render_target(params.out_w, params.out_h, &mut post.images);
        post.rt.0 = new_rt.clone();
        if let Ok(mut rt_comp) = post.cam_targets.get_mut(post.scene_cam.0) {
            *rt_comp = RenderTarget::Image(new_rt.clone().into());
        }
        if let Some(mat) = post.blur_mat.get_mut(&post.blur_h.0) {
            mat.source = new_rt;
            mat.uniforms.inv_w = 1.0 / params.out_w as f32;
            mat.uniforms.inv_h = 1.0 / params.out_h as f32;
        }
        if let Some(mesh) = meshes.get_mut(&post.blur_mesh.0) {
            *mesh = quad;
        }
    }

    // Rebuild tessellation mesh
    if dirty
        && matches!(
            params.style,
            Style::LowPoly | Style::Voronoi | Style::Penrose
        )
    {
        let new_mesh = dispatch_mesh(&params);
        for (mesh2d, _) in lp_query.iter() {
            if let Some(mesh) = meshes.get_mut(&mesh2d.0) {
                *mesh = new_mesh.clone();
            }
        }
    }

    if ht_dirty && let Some(mat) = mats.ht.get_mut(&handles.ht_mat.0) {
        mat.uniforms = params.to_ht_uniforms();
    }
    if tp_dirty && let Some(mat) = mats.tp.get_mut(&handles.tp_mat.0) {
        mat.uniforms = params.to_topo_uniforms();
    }
    if ms_dirty && let Some(mat) = mats.ms.get_mut(&handles.ms_mat.0) {
        mat.uniforms = params.to_mosaic_uniforms();
    }
    if eg_dirty && let Some(mat) = mats.eg.get_mut(&handles.eg_mat.0) {
        mat.uniforms = params.to_engrave_uniforms();
    }
    if wfc_grid_dirty {
        let new_grid = run_wfc(
            params.wfc_grid_w,
            params.wfc_grid_h,
            wfc_tiles(params.wfc_tile_set),
            params.seed,
        );
        params.wfc_grid = new_grid;
    }
    if (wfc_grid_dirty || wfc_vis_dirty) && let Some(mat) = mats.wfc.get_mut(&handles.wfc_mat.0) {
        mat.uniforms = params.to_wfc_uniforms();
    }
    if ck_dirty && let Some(mat) = mats.ck.get_mut(&handles.ck_mat.0) {
        mat.uniforms = params.to_crackle_uniforms();
    }
    if ff_dirty && let Some(mat) = mats.ff.get_mut(&handles.ff_mat.0) {
        mat.uniforms = params.to_ff_uniforms();
    }
    if oa_dirty && let Some(mat) = mats.oa.get_mut(&handles.oa_mat.0) {
        mat.uniforms = params.to_oa_uniforms();
    }
    if wv_dirty && let Some(mat) = mats.wv.get_mut(&handles.wv_mat.0) {
        mat.uniforms = params.to_wv_uniforms();
    }
    if sa_dirty && let Some(mat) = mats.sa.get_mut(&handles.sa_mat.0) {
        mat.uniforms = params.to_sa_uniforms();
    }
    if rd_dirty && let Some(mat) = mats.rd.get_mut(&handles.rd_mat.0) {
        mat.uniforms = params.to_rd_uniforms();
    }
    if mz_grid_dirty {
        params.mz_grid = run_maze(params.mz_grid_w, params.mz_grid_h, params.seed);
    }
    if (mz_grid_dirty || mz_vis_dirty) && let Some(mat) = mats.mz.get_mut(&handles.mz_mat.0) {
        mat.uniforms = params.to_mz_uniforms();
    }
    if cb_dirty && let Some(mat) = mats.cb.get_mut(&handles.cb_mat.0) {
        mat.uniforms = params.to_cb_uniforms();
    }
    if bp_dirty && let Some(mat) = mats.bp.get_mut(&handles.bp_mat.0) {
        mat.uniforms = params.to_bp_uniforms();
    }
    if blur_dirty && let Some(mat) = post.blur_mat.get_mut(&post.blur_h.0) {
        mat.uniforms.radius = params.blur_radius;
    }

    // Sync visibility: only one style visible at a time
    let style = params.style;
    for (_, mut vis) in lp_query.iter_mut() {
        *vis = if matches!(style, Style::LowPoly | Style::Voronoi | Style::Penrose) {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
    for (mut vis, id) in sv.q.iter_mut() {
        *vis = if id.0 == style {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }

    Ok(())
}

// ── UI helpers ────────────────────────────────────────────────────────────────

fn lp_slider(
    ui: &mut egui::Ui,
    label: &str,
    v: &mut usize,
    r: std::ops::RangeInclusive<usize>,
    d: &mut bool,
) {
    ui.label(label);
    *d |= ui.add(egui::Slider::new(v, r)).changed();
    ui.end_row();
}
fn fp_slider(
    ui: &mut egui::Ui,
    label: &str,
    v: &mut f32,
    r: std::ops::RangeInclusive<f32>,
    d: &mut bool,
) {
    ui.label(label);
    *d |= ui.add(egui::Slider::new(v, r)).changed();
    ui.end_row();
}
fn fp_slider_suffix(
    ui: &mut egui::Ui,
    label: &str,
    v: &mut f32,
    r: std::ops::RangeInclusive<f32>,
    suffix: &str,
    d: &mut bool,
) {
    ui.label(label);
    *d |= ui.add(egui::Slider::new(v, r).suffix(suffix)).changed();
    ui.end_row();
}
fn color_row(ui: &mut egui::Ui, label: &str, v: &mut [f32; 3], d: &mut bool) {
    ui.label(label);
    *d |= ui.color_edit_button_rgb(v).changed();
    ui.end_row();
}

// ── Dispatch ─────────────────────────────────────────────────────────────────

fn dispatch_mesh(params: &Params) -> Mesh {
    match params.style {
        Style::LowPoly => build_lowpoly(params),
        Style::Voronoi => build_voronoi(params),
        Style::Penrose => build_penrose(params),
        _ => Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        ),
    }
}

// ── Low-poly mesh ─────────────────────────────────────────────────────────────

fn build_lowpoly(p: &Params) -> Mesh {
    let (cols, rows, jitter) = (p.lp_cols, p.lp_rows, p.lp_jitter);
    let (w, h) = (p.out_w as f32, p.out_h as f32);
    let mut rng = StdRng::seed_from_u64(p.seed);
    let (cw, ch) = (w / cols as f32, h / rows as f32);

    let mut pts = vec![[0.0f32; 2]; (cols + 1) * (rows + 1)];
    for row in 0..=rows {
        for col in 0..=cols {
            let bx = col as f32 * cw - w * 0.5;
            let by = row as f32 * ch - h * 0.5;
            let jx = if col > 0 && col < cols {
                rng.gen_range(-0.5..0.5) * jitter * cw
            } else {
                0.0
            };
            let jy = if row > 0 && row < rows {
                rng.gen_range(-0.5..0.5) * jitter * ch
            } else {
                0.0
            };
            pts[row * (cols + 1) + col] = [bx + jx, by + jy];
        }
    }

    let pt = |r: usize, c: usize| pts[r * (cols + 1) + c];
    let mut pos: Vec<[f32; 3]> = Vec::with_capacity(cols * rows * 6);
    let mut col: Vec<[f32; 4]> = Vec::with_capacity(cols * rows * 6);

    for row in 0..rows {
        for col_ in 0..cols {
            let p00 = pt(row, col_);
            let p10 = pt(row, col_ + 1);
            let p01 = pt(row + 1, col_);
            let p11 = pt(row + 1, col_ + 1);
            lp_tri(&mut pos, &mut col, p00, p10, p11, p, &mut rng, w, h);
            lp_tri(&mut pos, &mut col, p00, p11, p01, p, &mut rng, w, h);
        }
    }

    mesh_from(pos, col)
}

#[allow(clippy::too_many_arguments)]
fn lp_tri(
    pos: &mut Vec<[f32; 3]>,
    col: &mut Vec<[f32; 4]>,
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
    p: &Params,
    rng: &mut StdRng,
    w: f32,
    h: f32,
) {
    let cx = (a[0] + b[0] + c[0]) / 3.0;
    let cy = (a[1] + b[1] + c[1]) / 3.0;
    let [r, g, bv, _] = bilinear(cx, cy, w, h, p.lp_c_tl, p.lp_c_tr, p.lp_c_bl, p.lp_c_br);
    let v = rng.gen_range(-p.lp_color_var..p.lp_color_var);
    let clr = [
        (r + v).clamp(0., 1.),
        (g + v).clamp(0., 1.),
        (bv + v).clamp(0., 1.),
        1.0,
    ];
    for pt in [a, b, c] {
        pos.push([pt[0], pt[1], 0.]);
        col.push(clr);
    }
}

// ── Voronoi mesh ──────────────────────────────────────────────────────────────

fn build_voronoi(p: &Params) -> Mesh {
    let (cols, rows, jitter) = (p.vor_cols, p.vor_rows, p.vor_jitter);
    let (w, h) = (p.out_w as f32, p.out_h as f32);
    let mut rng = StdRng::seed_from_u64(p.seed);
    let (cw, ch) = (w / cols as f32, h / rows as f32);

    let nc = cols + 1;
    let nr = rows + 1;
    let mut seeds = vec![[0.0f32; 2]; nc * nr];
    for row in 0..nr {
        for col in 0..nc {
            let bx = col as f32 * cw - w * 0.5;
            let by = row as f32 * ch - h * 0.5;
            let jx = if col > 0 && col < nc - 1 {
                rng.gen_range(-0.5..0.5) * jitter * cw
            } else {
                0.
            };
            let jy = if row > 0 && row < nr - 1 {
                rng.gen_range(-0.5..0.5) * jitter * ch
            } else {
                0.
            };
            seeds[row * nc + col] = [bx + jx, by + jy];
        }
    }

    let mut pos: Vec<[f32; 3]> = Vec::new();
    let mut col: Vec<[f32; 4]> = Vec::new();

    for row in 0..nr {
        for c in 0..nc {
            let sp = seeds[row * nc + c];
            let poly = voronoi_cell(&seeds, row, c, nr, nc, w, h);
            if poly.len() < 3 {
                continue;
            }

            let centroid = poly_centroid(&poly);
            let inset = inset_poly(&poly, centroid, p.vor_border);
            if inset.len() < 3 || poly_area(&inset).abs() < 4.0 {
                continue;
            }

            let [r, g, bv, _] = bilinear(
                sp[0], sp[1], w, h, p.vor_c_tl, p.vor_c_tr, p.vor_c_bl, p.vor_c_br,
            );
            let v = rng.gen_range(-p.vor_color_var..p.vor_color_var);
            let base = [
                (r + v).clamp(0., 1.),
                (g + v).clamp(0., 1.),
                (bv + v).clamp(0., 1.),
                1.0_f32,
            ];
            let e = p.vor_elevation;
            let hi = [
                (base[0] + e).clamp(0., 1.),
                (base[1] + e).clamp(0., 1.),
                (base[2] + e).clamp(0., 1.),
                1.0,
            ];

            let n = inset.len();
            for i in 0..n {
                let a = inset[i];
                let b = inset[(i + 1) % n];
                pos.push([centroid[0], centroid[1], 0.]);
                col.push(hi);
                pos.push([a[0], a[1], 0.]);
                col.push(base);
                pos.push([b[0], b[1], 0.]);
                col.push(base);
            }
        }
    }

    mesh_from(pos, col)
}

fn voronoi_cell(
    seeds: &[[f32; 2]],
    row: usize,
    col: usize,
    nr: usize,
    nc: usize,
    w: f32,
    h: f32,
) -> Vec<[f32; 2]> {
    let seed = seeds[row * nc + col];
    let pad = 50.0;
    let mut poly = vec![
        [-w * 0.5 - pad, -h * 0.5 - pad],
        [w * 0.5 + pad, -h * 0.5 - pad],
        [w * 0.5 + pad, h * 0.5 + pad],
        [-w * 0.5 - pad, h * 0.5 + pad],
    ];
    for dr in -1i32..=1 {
        for dc in -1i32..=1 {
            if dr == 0 && dc == 0 {
                continue;
            }
            let nr2 = row as i32 + dr;
            let nc2 = col as i32 + dc;
            if nr2 < 0 || nr2 >= nr as i32 || nc2 < 0 || nc2 >= nc as i32 {
                continue;
            }
            let nbr = seeds[nr2 as usize * nc + nc2 as usize];
            poly = clip_halfplane(poly, seed, nbr);
            if poly.is_empty() {
                return poly;
            }
        }
    }
    poly
}

fn clip_halfplane(poly: Vec<[f32; 2]>, seed: [f32; 2], nbr: [f32; 2]) -> Vec<[f32; 2]> {
    if poly.is_empty() {
        return poly;
    }
    let mx = (seed[0] + nbr[0]) * 0.5;
    let my = (seed[1] + nbr[1]) * 0.5;
    let nx = seed[0] - nbr[0];
    let ny = seed[1] - nbr[1];
    let dot = |p: [f32; 2]| (p[0] - mx) * nx + (p[1] - my) * ny;
    let isect = |a: [f32; 2], b: [f32; 2]| -> [f32; 2] {
        let da = dot(a);
        let db = dot(b);
        let t = da / (da - db);
        [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
    };
    let n = poly.len();
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let ai = dot(a) >= 0.0;
        let bi = dot(b) >= 0.0;
        if ai {
            out.push(a);
            if !bi {
                out.push(isect(a, b));
            }
        } else if bi {
            out.push(isect(a, b));
        }
    }
    out
}

fn poly_centroid(poly: &[[f32; 2]]) -> [f32; 2] {
    let n = poly.len() as f32;
    [
        poly.iter().map(|p| p[0]).sum::<f32>() / n,
        poly.iter().map(|p| p[1]).sum::<f32>() / n,
    ]
}

fn inset_poly(poly: &[[f32; 2]], c: [f32; 2], amount: f32) -> Vec<[f32; 2]> {
    poly.iter()
        .map(|p| {
            let dx = p[0] - c[0];
            let dy = p[1] - c[1];
            let d = (dx * dx + dy * dy).sqrt();
            if d < 1e-4 {
                c
            } else {
                let t = (amount / d).min(0.95);
                [p[0] - dx * t, p[1] - dy * t]
            }
        })
        .collect()
}

fn poly_area(poly: &[[f32; 2]]) -> f32 {
    let n = poly.len();
    let mut a = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        a += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    a * 0.5
}

// ── WFC algorithm ─────────────────────────────────────────────────────────────
//
// Tile bitmask: bit0=top, bit1=right, bit2=bottom, bit3=left
//  V=5(0101), H=10(1010), TR=3(0011), BR=6(0110), BL=12(1100), TL=9(1001)
//  Cross=15(1111), T_TRB=7(0111), T_RBL=14(1110), T_TBL=13(1101), T_TRL=11(1011)

fn wfc_tiles(tile_set: u8) -> &'static [u8] {
    match tile_set {
        0 => &[5, 10, 3, 6, 12, 9],
        1 => &[5, 10, 3, 6, 12, 9, 15],
        _ => &[5, 10, 3, 6, 12, 9, 15, 7, 14, 13, 11],
    }
}

fn wfc_compatible(a: u8, b: u8, dx: i32, dy: i32) -> bool {
    match (dx, dy) {
        (1, 0) => (a >> 1 & 1) == (b >> 3 & 1),  // a.right == b.left
        (-1, 0) => (a >> 3 & 1) == (b >> 1 & 1), // a.left  == b.right
        (0, 1) => (a >> 2 & 1) == (b & 1),        // a.bottom == b.top
        (0, -1) => (a & 1) == (b >> 2 & 1),       // a.top   == b.bottom
        _ => true,
    }
}

fn run_wfc(w: usize, h: usize, tiles: &[u8], seed: u64) -> Vec<u8> {
    for attempt in 0..20u64 {
        if let Some(grid) = try_wfc(w, h, tiles, seed.wrapping_add(attempt * 7919)) {
            return grid;
        }
    }
    // Fallback: checkerboard H/V
    let h_t = if tiles.contains(&10) { 10 } else { tiles[0] };
    let v_t = if tiles.contains(&5) { 5 } else { tiles[0] };
    (0..w * h)
        .map(|i| if (i / w + i % w).is_multiple_of(2) { h_t } else { v_t })
        .collect()
}

fn try_wfc(w: usize, h: usize, tiles: &[u8], seed: u64) -> Option<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = w * h;
    let dirs: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    let mut wave: Vec<Vec<u8>> = vec![tiles.to_vec(); n];

    for _ in 0..n {
        let cell = (0..n)
            .filter(|&i| wave[i].len() > 1)
            .min_by_key(|&i| wave[i].len());
        let Some(cell) = cell else { break };
        if wave[cell].is_empty() {
            return None;
        }

        let idx = rng.gen_range(0..wave[cell].len());
        let chosen = wave[cell][idx];
        wave[cell] = vec![chosen];

        let mut stack = vec![cell];
        let mut contradiction = false;
        while let Some(cur) = stack.pop() {
            let x = (cur % w) as i32;
            let y = (cur / w) as i32;
            for &(dx, dy) in &dirs {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let ncur = ny as usize * w + nx as usize;
                if wave[ncur].len() <= 1 {
                    continue;
                }
                let cur_opts = wave[cur].clone();
                let before = wave[ncur].len();
                wave[ncur].retain(|&nb| cur_opts.iter().any(|&cb| wfc_compatible(cb, nb, dx, dy)));
                if wave[ncur].is_empty() {
                    contradiction = true;
                    break;
                }
                if wave[ncur].len() < before {
                    stack.push(ncur);
                }
            }
            if contradiction {
                break;
            }
        }
        if contradiction {
            return None;
        }
    }

    Some(
        wave.into_iter()
            .map(|opts| opts.into_iter().next().unwrap_or(tiles[0]))
            .collect(),
    )
}

// ── Shared ────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn bilinear(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    c_tl: [f32; 3],
    c_tr: [f32; 3],
    c_bl: [f32; 3],
    c_br: [f32; 3],
) -> [f32; 4] {
    let tx = (x / w + 0.5).clamp(0., 1.);
    let ty = (y / h + 0.5).clamp(0., 1.);
    let lrp = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]
    };
    let [r, g, b] = lrp(lrp(c_bl, c_br, tx), lrp(c_tl, c_tr, tx), ty);
    [r, g, b, 1.0]
}

fn mesh_from(positions: Vec<[f32; 3]>, colors: Vec<[f32; 4]>) -> Mesh {
    let mut m = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    m.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    m.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    m
}

// ── Maze algorithm ─────────────────────────────────────────────────────────────
//
// Wall bitmask per cell: bit0=top, bit1=right, bit2=bottom, bit3=left
// Recursive backtracker: start with all walls, carve passages.

fn run_maze(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut walls = vec![0xfu8; w * h]; // all walls present
    let mut visited = vec![false; w * h];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut stack = vec![(0usize, 0usize)];
    visited[0] = true;

    // (dx, dy, wall_bit_from, wall_bit_to)
    let dirs: [(i32, i32, u8, u8); 4] = [
        (0, -1, 0x1, 0x4), // up:    remove top of current, bottom of neighbor
        (1, 0, 0x2, 0x8),  // right: remove right of current, left of neighbor
        (0, 1, 0x4, 0x1),  // down:  remove bottom of current, top of neighbor
        (-1, 0, 0x8, 0x2), // left:  remove left of current, right of neighbor
    ];

    while let Some(&(cx, cy)) = stack.last() {
        let neighbors: Vec<_> = dirs
            .iter()
            .filter_map(|&(dx, dy, wf, wt)| {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                    let ni = ny as usize * w + nx as usize;
                    if !visited[ni] {
                        return Some((nx as usize, ny as usize, wf, wt));
                    }
                }
                None
            })
            .collect();

        if neighbors.is_empty() {
            stack.pop();
        } else {
            let &(nx, ny, wf, wt) = &neighbors[rng.gen_range(0..neighbors.len())];
            walls[cy * w + cx] &= !wf;
            walls[ny * w + nx] &= !wt;
            visited[ny * w + nx] = true;
            stack.push((nx, ny));
        }
    }
    walls
}

// ── Penrose P2 tiling ──────────────────────────────────────────────────────────
//
// Uses Robinson-triangle deflation (P2 kite/dart tiling).
// Triangle types:
//   Red  (R): isoceles, apex 36°, apex at v[0], legs = φ, base = 1
//   Blue (B): isoceles, apex 108°, apex at v[0], legs = 1, base = φ
// Deflation:
//   R(a,b,c) → R(c,p,b) + B(p,c,a)   where p is on ab at distance |ac|=1 from a
//   B(a,b,c) → B(d,a,c) + B(d,b,a) + R(d,b,c)  where d is on bc at dist 1 from b
// Start: 10 red triangles in a wheel (sun pattern).

#[derive(Clone, Copy)]
enum PTriKind {
    Red,
    Blue,
}

#[derive(Clone)]
struct PTri {
    kind: PTriKind,
    v: [Vec2; 3],
}

fn deflate(tris: Vec<PTri>) -> Vec<PTri> {
    let phi = (1.0_f32 + 5.0_f32.sqrt()) * 0.5;
    let mut out = Vec::with_capacity(tris.len() * 3);
    for t in tris {
        let [a, b, c] = t.v;
        match t.kind {
            PTriKind::Red => {
                // p on AB such that |AP| = 1/φ (= |AC|)
                let p = a + (b - a) / phi;
                out.push(PTri {
                    kind: PTriKind::Red,
                    v: [c, p, b],
                });
                out.push(PTri {
                    kind: PTriKind::Blue,
                    v: [p, c, a],
                });
            }
            PTriKind::Blue => {
                // q on BA, r on BC
                let q = b + (a - b) / phi;
                let r = b + (c - b) / phi;
                out.push(PTri {
                    kind: PTriKind::Blue,
                    v: [r, c, a],
                });
                out.push(PTri {
                    kind: PTriKind::Blue,
                    v: [q, r, b],
                });
                out.push(PTri {
                    kind: PTriKind::Red,
                    v: [r, q, a],
                });
            }
        }
    }
    out
}

fn build_penrose(p: &Params) -> Mesh {
    let phi = (1.0_f32 + 5.0_f32.sqrt()) * 0.5;
    let (w, h) = (p.out_w as f32, p.out_h as f32);
    let radius = w.max(h) * 0.65;

    // Initial sun: 10 red triangles around origin
    let mut tris: Vec<PTri> = (0..10)
        .map(|i| {
            let a0 = std::f32::consts::PI * (2 * i) as f32 / 10.0;
            let a1 = std::f32::consts::PI * (2 * i + 2) as f32 / 10.0;
            let kind = if i % 2 == 0 {
                PTriKind::Red
            } else {
                // Alternate chirality so the tiling tiles correctly
                PTriKind::Red
            };
            // Apex at center, two base vertices on outer ring
            let mut v = [
                Vec2::ZERO,
                Vec2::new(a0.cos(), a0.sin()) * radius,
                Vec2::new(a1.cos(), a1.sin()) * radius,
            ];
            // For correct P2 tiling, every other triangle needs flipped winding
            if i % 2 != 0 {
                v.swap(1, 2);
            }
            PTri { kind, v }
        })
        .collect();

    // Scale the initial triangles: apex sides should be φ * base, and base = 2*radius*sin(18°)
    // The radius is already set; just adjust so |apex_side| = φ * |base|:
    // For a regular setup, scale the outer ring by φ so apex sides = φ * outer_step
    let scale = phi;
    for t in &mut tris {
        t.v[1] *= scale;
        t.v[2] *= scale;
    }

    for _ in 0..p.pe_iterations {
        tris = deflate(tris);
    }

    // Build triangle mesh, clipping to viewport bounds
    let half_w = w * 0.5;
    let half_h = h * 0.5;
    let mut pos: Vec<[f32; 3]> = Vec::new();
    let mut col: Vec<[f32; 4]> = Vec::new();
    let mut rng = StdRng::seed_from_u64(p.seed);

    for t in &tris {
        // Clip: skip if all 3 vertices are outside on the same side
        let xs = [t.v[0].x, t.v[1].x, t.v[2].x];
        let ys = [t.v[0].y, t.v[1].y, t.v[2].y];
        if xs.iter().all(|&x| x < -half_w) || xs.iter().all(|&x| x > half_w) {
            continue;
        }
        if ys.iter().all(|&y| y < -half_h) || ys.iter().all(|&y| y > half_h) {
            continue;
        }

        let base = match t.kind {
            PTriKind::Red => p.pe_color_a,
            PTriKind::Blue => p.pe_color_b,
        };
        let v = rng.gen_range(-p.pe_color_var..p.pe_color_var);
        let c = [
            (base[0] + v).clamp(0., 1.),
            (base[1] + v).clamp(0., 1.),
            (base[2] + v).clamp(0., 1.),
            1.0_f32,
        ];
        for vi in &t.v {
            pos.push([vi.x, vi.y, 0.0]);
            col.push(c);
        }
    }

    mesh_from(pos, col)
}

fn make_render_target(w: u32, h: u32, images: &mut Assets<Image>) -> Handle<Image> {
    let size = Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    let mut img = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
    images.add(img)
}
