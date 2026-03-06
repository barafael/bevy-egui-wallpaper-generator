use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use rand::{Rng, SeedableRng, rngs::StdRng};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::fmt;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen(start))]
pub fn start() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "RandomArt  [Space] = new".into(),
                    resolution: (800u32, 600u32).into(),
                    fit_canvas_to_parent: true,
                    ..default()
                }),
                ..default()
            }),
            EguiPlugin::default(),
        ))
        .init_resource::<Params>()
        .add_systems(Startup, setup)
        .add_systems(EguiPrimaryContextPass, ui)
        .add_systems(Update, (handle_regen, animate, fit_to_window).chain())
        .run();
}

// ─── Params ───────────────────────────────────────────────────────────────────

#[derive(Resource)]
struct Params {
    depth: u32,
    anim_speed: f32,
    img_w: u32,
    img_h: u32,
    w_terminal: u32,
    w_add: u32,
    w_mult: u32,
    w_sqrt: u32,
    w_sin: u32,
    w_mod: u32,
    w_mix: u32,
    needs_regen: bool,
}

impl Default for Params {
    fn default() -> Self {
        // Start at half resolution on WASM; single-threaded eval is ~4× slower
        #[cfg(target_arch = "wasm32")]
        let (img_w, img_h) = (400u32, 300u32);
        #[cfg(not(target_arch = "wasm32"))]
        let (img_w, img_h) = (800u32, 600u32);

        Self {
            depth: 7,
            anim_speed: 0.4,
            img_w,
            img_h,
            w_terminal: 2,
            w_add: 3,
            w_mult: 3,
            w_sqrt: 1,
            w_sin: 2,
            w_mod: 1,
            w_mix: 1,
            needs_regen: false,
        }
    }
}

// ─── State ───────────────────────────────────────────────────────────────────

#[derive(Component)]
struct ArtSprite;

#[derive(Resource)]
struct ArtState {
    r: Expr,
    g: Expr,
    b: Expr,
    // Flat bytecode programs — cache-friendly, no heap-pointer chasing per pixel
    r_prog: Vec<Op>,
    g_prog: Vec<Op>,
    b_prog: Vec<Op>,
    image: Handle<Image>,
    img_w: u32,
    img_h: u32,
}

// ─── Systems ─────────────────────────────────────────────────────────────────

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>, params: Res<Params>) {
    commands.spawn(Camera2d);
    let seed: u64 = rand::thread_rng().r#gen();
    let (state, handle) = new_art(seed, &params, &mut images);
    commands.insert_resource(state);
    commands.spawn((Sprite { image: handle, ..default() }, ArtSprite));
}

fn handle_regen(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut params: ResMut<Params>,
    query: Query<Entity, With<ArtSprite>>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        params.needs_regen = true;
    }
    if !params.needs_regen {
        return;
    }
    params.needs_regen = false;

    for e in &query {
        commands.entity(e).despawn();
    }
    let seed: u64 = rand::thread_rng().r#gen();
    let (state, handle) = new_art(seed, &params, &mut images);
    commands.insert_resource(state);
    commands.spawn((Sprite { image: handle, ..default() }, ArtSprite));
}

fn ui(
    mut contexts: EguiContexts,
    mut params: ResMut<Params>,
    art: Option<Res<ArtState>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut style_done: Local<bool>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    // Apply style once — egui Context persists across frames, no need to repeat
    if !*style_done {
        apply_style(ctx);
        *style_done = true;
    }

    const YELLOW: egui::Color32 = egui::Color32::from_rgb(255, 224, 102);
    let hd = |s: &str| egui::RichText::new(s).monospace().strong().color(YELLOW);
    let lbl = |s: &str| egui::RichText::new(s).monospace();

    egui::Window::new(
        egui::RichText::new("// RANDOMFART")
            .monospace()
            .strong()
            .color(egui::Color32::WHITE),
    )
    .default_pos([10.0, 10.0])
    .resizable(false)
    .show(ctx, |ui| {
        // ── Tree ────────────────────────────────────────────────────────
        ui.label(hd("[ TREE ]"));
        let prev = params.depth;
        ui.add(fat_slider(
            egui::Slider::new(&mut params.depth, 1..=10).text(lbl("DEPTH")),
        ));
        if params.depth != prev {
            params.needs_regen = true;
        }

        ui.add_space(4.0);
        ui.label(hd("[ GRAMMAR WEIGHTS ]"));
        let prev = (
            params.w_terminal,
            params.w_add,
            params.w_mult,
            params.w_sqrt,
            params.w_sin,
            params.w_mod,
            params.w_mix,
        );
        ui.add(fat_slider(
            egui::Slider::new(&mut params.w_terminal, 1..=8).text(lbl("TERMINAL")),
        ));
        ui.add(fat_slider(
            egui::Slider::new(&mut params.w_add, 1..=8).text(lbl("ADD")),
        ));
        ui.add(fat_slider(
            egui::Slider::new(&mut params.w_mult, 1..=8).text(lbl("MULT")),
        ));
        ui.add(fat_slider(
            egui::Slider::new(&mut params.w_sqrt, 1..=8).text(lbl("SQRT(ABS)")),
        ));
        ui.add(fat_slider(
            egui::Slider::new(&mut params.w_sin, 1..=8).text(lbl("SIN")),
        ));
        ui.add(fat_slider(
            egui::Slider::new(&mut params.w_mod, 1..=8).text(lbl("MOD")),
        ));
        ui.add(fat_slider(
            egui::Slider::new(&mut params.w_mix, 1..=8).text(lbl("MIX")),
        ));
        if (
            params.w_terminal,
            params.w_add,
            params.w_mult,
            params.w_sqrt,
            params.w_sin,
            params.w_mod,
            params.w_mix,
        ) != prev
        {
            params.needs_regen = true;
        }

        ui.separator();

        // ── Animation ───────────────────────────────────────────────────
        ui.label(hd("[ ANIMATION ]"));
        ui.add(fat_slider(
            egui::Slider::new(&mut params.anim_speed, 0.0..=3.0)
                .text(lbl("SPEED"))
                .step_by(0.05),
        ));

        ui.separator();

        // ── Resolution ──────────────────────────────────────────────────
        ui.label(hd("[ RESOLUTION ]"));
        const PRESETS: &[(&str, u32, u32)] = &[
            ("POTATO  200x150", 200, 150),
            ("LOW     400x300", 400, 300),
            ("MED     800x600", 800, 600),
            ("HIGH   1280x960", 1280, 960),
        ];
        let current_label = PRESETS
            .iter()
            .find(|&&(_, w, h)| w == params.img_w && h == params.img_h)
            .map(|&(s, _, _)| s)
            .unwrap_or("CUSTOM");
        egui::ComboBox::from_id_salt("resolution")
            .selected_text(egui::RichText::new(current_label).monospace())
            .show_ui(ui, |ui| {
                for &(label, w, h) in PRESETS {
                    if ui
                        .selectable_label(
                            params.img_w == w && params.img_h == h,
                            egui::RichText::new(label).monospace(),
                        )
                        .clicked()
                    {
                        params.img_w = w;
                        params.img_h = h;
                        params.needs_regen = true;
                    }
                }
            });

        ui.separator();

        let btn = egui::RichText::new("▶  NEW FART  [SPACE]")
            .monospace()
            .strong()
            .color(egui::Color32::WHITE);
        if ui.button(btn).clicked() {
            params.needs_regen = true;
        }
    });

    let ctrl = keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight);
    if ctrl && let Some(art) = &art {
        egui::Window::new(
            egui::RichText::new("// FORMULA")
                .monospace()
                .strong()
                .color(egui::Color32::WHITE),
        )
        .default_pos([10.0, 500.0])
        .resizable(true)
        .default_width(500.0)
        .show(ctx, |ui| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
            for (ch, expr) in [("R", &art.r), ("G", &art.g), ("B", &art.b)] {
                ui.label(hd(ch));
                ui.label(egui::RichText::new(expr.to_string()).monospace().size(10.0));
                ui.add_space(4.0);
            }
        });
    }

    Ok(())
}

fn fat_slider(s: egui::Slider) -> egui::Slider {
    s.handle_shape(egui::style::HandleShape::Rect { aspect_ratio: 0.6 })
}

fn apply_style(ctx: &egui::Context) {
    use egui::{Color32, CornerRadius, FontId, Shadow, Stroke};

    let bg = Color32::from_rgb(15, 15, 15);
    let panel = Color32::from_rgb(22, 22, 22);
    let white = Color32::WHITE;
    let yellow = Color32::from_rgb(255, 224, 102);
    let zero = CornerRadius::ZERO;
    let s2w = Stroke::new(2.0, white);
    let s2y = Stroke::new(2.0, yellow);

    let mut v = egui::Visuals::dark();
    v.override_text_color = Some(white);
    v.window_fill = panel;
    v.panel_fill = bg;
    v.window_corner_radius = zero;
    v.window_stroke = s2w;
    v.popup_shadow = Shadow {
        offset: [0, 0],
        blur: 0,
        spread: 0,
        color: Color32::TRANSPARENT,
    };
    v.window_shadow = Shadow {
        offset: [4, 4],
        blur: 0,
        spread: 0,
        color: yellow,
    };
    v.slider_trailing_fill = true;
    v.selection.bg_fill = Color32::from_rgb(180, 130, 0);
    v.selection.stroke = s2y;

    v.widgets.noninteractive.corner_radius = zero;
    v.widgets.noninteractive.bg_fill = panel;
    v.widgets.noninteractive.weak_bg_fill = panel;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, Color32::from_gray(55));
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, Color32::from_gray(160));

    v.widgets.inactive.corner_radius = zero;
    v.widgets.inactive.bg_fill = Color32::from_gray(55);
    v.widgets.inactive.weak_bg_fill = Color32::from_gray(55);
    v.widgets.inactive.bg_stroke = s2w;
    v.widgets.inactive.fg_stroke = s2w;

    v.widgets.hovered.corner_radius = zero;
    v.widgets.hovered.bg_fill = Color32::from_rgb(38, 38, 38);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(38, 38, 38);
    v.widgets.hovered.bg_stroke = s2y;
    v.widgets.hovered.fg_stroke = s2y;

    v.widgets.active.corner_radius = zero;
    v.widgets.active.bg_fill = yellow;
    v.widgets.active.weak_bg_fill = yellow;
    v.widgets.active.bg_stroke = s2w;
    v.widgets.active.fg_stroke = Stroke::new(2.0, bg);

    v.widgets.open.corner_radius = zero;
    v.widgets.open.bg_fill = Color32::from_rgb(30, 30, 30);
    v.widgets.open.weak_bg_fill = Color32::from_rgb(30, 30, 30);
    v.widgets.open.bg_stroke = s2y;
    v.widgets.open.fg_stroke = s2y;

    ctx.set_visuals(v);

    let mut s = (*ctx.style()).clone();
    s.override_font_id = Some(FontId::monospace(12.0));
    s.spacing.item_spacing = egui::vec2(8.0, 6.0);
    s.spacing.button_padding = egui::vec2(12.0, 8.0);
    s.spacing.interact_size = egui::vec2(40.0, 28.0);
    s.spacing.slider_width = 160.0;
    ctx.set_style(s);
}

fn animate(
    mut images: ResMut<Assets<Image>>,
    art: Option<Res<ArtState>>,
    params: Res<Params>,
    time: Res<Time>,
    // Local state persists across frames without touching ArtState's change detection
    mut pixel_buf: Local<Vec<u8>>,
    mut last_t: Local<f32>,
) {
    let Some(art) = art else { return };

    let t = (time.elapsed_secs() * params.anim_speed).sin();

    // Skip render when t is unchanged (only meaningful when anim_speed = 0) or
    // when new art was just inserted (is_changed() forces the first render).
    if !art.is_changed() && (t - *last_t).abs() < 1e-5 {
        return;
    }
    *last_t = t;

    let (img_w, img_h) = (art.img_w, art.img_h);
    let size = (img_w * img_h * 4) as usize;

    // Reuse buffer across frames; reallocate only when resolution changes.
    // Alpha channel is always 255 — we never overwrite chunk[3].
    if pixel_buf.len() != size {
        *pixel_buf = vec![255u8; size];
    }

    let r = art.r_prog.as_slice();
    let g = art.g_prog.as_slice();
    let b = art.b_prog.as_slice();

    let eval = |(i, chunk): (usize, &mut [u8])| {
        let i = i as u32;
        let py = (i / img_w) as f32 / img_h as f32 * 2.0 - 1.0;
        let px = (i % img_w) as f32 / img_w as f32 * 2.0 - 1.0;
        chunk[0] = channel(eval_program(r, px, py, t));
        chunk[1] = channel(eval_program(g, px, py, t));
        chunk[2] = channel(eval_program(b, px, py, t));
    };

    #[cfg(not(target_arch = "wasm32"))]
    pixel_buf.par_chunks_mut(4).enumerate().for_each(eval);
    #[cfg(target_arch = "wasm32")]
    pixel_buf.chunks_mut(4).enumerate().for_each(eval);

    // Write into the image's existing Vec when possible (avoids reallocation).
    let handle = art.image.clone();
    if let Some(image) = images.get_mut(&handle) {
        match &mut image.data {
            Some(data) if data.len() == size => data.copy_from_slice(&pixel_buf),
            slot => *slot = Some(pixel_buf.clone()),
        }
    }
}

fn fit_to_window(window: Query<&Window>, mut sprite: Query<&mut Sprite, With<ArtSprite>>) {
    let Ok(window) = window.single() else { return };
    let size = Vec2::new(window.width(), window.height());
    for mut sprite in &mut sprite {
        sprite.custom_size = Some(size);
    }
}

// ─── Art generation ──────────────────────────────────────────────────────────

fn new_art(seed: u64, params: &Params, images: &mut Assets<Image>) -> (ArtState, Handle<Image>) {
    info!(
        "seed={seed}  depth={}  res={}×{}",
        params.depth, params.img_w, params.img_h
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let r = Expr::generate(&mut rng, params.depth, params);
    let g = Expr::generate(&mut rng, params.depth, params);
    let b = Expr::generate(&mut rng, params.depth, params);

    let mut r_prog = Vec::new();
    let mut g_prog = Vec::new();
    let mut b_prog = Vec::new();
    r.compile(&mut r_prog);
    g.compile(&mut g_prog);
    b.compile(&mut b_prog);

    let (img_w, img_h) = (params.img_w, params.img_h);
    let size = (img_w * img_h * 4) as usize;

    // Render t=0 immediately so the sprite isn't blank on its first frame.
    let mut pixels = vec![255u8; size];
    let eval = |(i, chunk): (usize, &mut [u8])| {
        let i = i as u32;
        let py = (i / img_w) as f32 / img_h as f32 * 2.0 - 1.0;
        let px = (i % img_w) as f32 / img_w as f32 * 2.0 - 1.0;
        chunk[0] = channel(eval_program(&r_prog, px, py, 0.0));
        chunk[1] = channel(eval_program(&g_prog, px, py, 0.0));
        chunk[2] = channel(eval_program(&b_prog, px, py, 0.0));
    };
    #[cfg(not(target_arch = "wasm32"))]
    pixels.par_chunks_mut(4).enumerate().for_each(eval);
    #[cfg(target_arch = "wasm32")]
    pixels.chunks_mut(4).enumerate().for_each(eval);

    let image = Image::new(
        Extent3d { width: img_w, height: img_h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    let handle = images.add(image);
    (
        ArtState { r, g, b, r_prog, g_prog, b_prog, image: handle.clone(), img_w, img_h },
        handle,
    )
}

// ─── Bytecode VM ─────────────────────────────────────────────────────────────

/// Flat stack-machine opcode. `Copy` so we can iterate by value.
#[derive(Clone, Copy)]
enum Op {
    X,
    Y,
    T,
    Num(f32),
    Add,
    Mult,
    Sqrt, // sqrt(abs(top))
    Abs,
    Sin, // sin(π·top)
    Mod, // normalized euclid-mod
    Mix, // mix(a, b; weight=c) — pops c, b, a
}

/// Evaluate a compiled program on the stack machine.
///
/// Stack depth is bounded by 2·depth+1 (worst case: all-Mix tree).
/// 32 slots covers depth ≤ 15 with room to spare.
fn eval_program(ops: &[Op], x: f32, y: f32, t: f32) -> f32 {
    let mut stack = [0f32; 32];
    let mut sp = 0usize;
    for &op in ops {
        match op {
            Op::X => {
                stack[sp] = x;
                sp += 1;
            }
            Op::Y => {
                stack[sp] = y;
                sp += 1;
            }
            Op::T => {
                stack[sp] = t;
                sp += 1;
            }
            Op::Num(n) => {
                stack[sp] = n;
                sp += 1;
            }
            Op::Abs => stack[sp - 1] = stack[sp - 1].abs(),
            Op::Sqrt => stack[sp - 1] = stack[sp - 1].abs().sqrt(),
            Op::Sin => stack[sp - 1] = (stack[sp - 1] * std::f32::consts::PI).sin(),
            Op::Add => {
                sp -= 1;
                stack[sp - 1] += stack[sp];
            }
            Op::Mult => {
                sp -= 1;
                stack[sp - 1] *= stack[sp];
            }
            Op::Mod => {
                sp -= 1;
                let bv = stack[sp].abs().max(0.001);
                stack[sp - 1] = stack[sp - 1].rem_euclid(bv) / bv * 2.0 - 1.0;
            }
            Op::Mix => {
                // stack: [..., a, b, c]  sp points past c
                // after sp-=2: stack[sp-1]=a, stack[sp]=b, stack[sp+1]=c
                sp -= 2;
                let w = ((stack[sp + 1] + 1.0) * 0.5).clamp(0.0, 1.0);
                stack[sp - 1] = stack[sp - 1] * (1.0 - w) + stack[sp] * w;
            }
        }
    }
    stack[0]
}

// ─── Expression tree ─────────────────────────────────────────────────────────

#[derive(Clone)]
enum Expr {
    X,
    Y,
    T,
    Num(f32),
    Add(Box<Expr>, Box<Expr>),
    Mult(Box<Expr>, Box<Expr>),
    Sqrt(Box<Expr>),
    Abs(Box<Expr>),
    Sin(Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),
    Mix(Box<Expr>, Box<Expr>, Box<Expr>),
}

impl Expr {
    /// Compile the expression tree to a flat postfix bytecode program.
    fn compile(&self, ops: &mut Vec<Op>) {
        match self {
            Expr::X => ops.push(Op::X),
            Expr::Y => ops.push(Op::Y),
            Expr::T => ops.push(Op::T),
            Expr::Num(n) => ops.push(Op::Num(*n)),
            Expr::Abs(e) => {
                e.compile(ops);
                ops.push(Op::Abs);
            }
            Expr::Sqrt(e) => {
                e.compile(ops);
                ops.push(Op::Sqrt);
            }
            Expr::Sin(e) => {
                e.compile(ops);
                ops.push(Op::Sin);
            }
            Expr::Add(a, b) => {
                a.compile(ops);
                b.compile(ops);
                ops.push(Op::Add);
            }
            Expr::Mult(a, b) => {
                a.compile(ops);
                b.compile(ops);
                ops.push(Op::Mult);
            }
            Expr::Mod(a, b) => {
                a.compile(ops);
                b.compile(ops);
                ops.push(Op::Mod);
            }
            Expr::Mix(a, b, c) => {
                a.compile(ops);
                b.compile(ops);
                c.compile(ops);
                ops.push(Op::Mix);
            }
        }
    }

    fn generate(rng: &mut StdRng, depth: u32, params: &Params) -> Self {
        let total = params.w_terminal
            + params.w_add
            + params.w_mult
            + params.w_sqrt
            + params.w_sin
            + params.w_mod
            + params.w_mix;
        let roll = rng.r#gen::<u32>() % total;
        let t_end = params.w_terminal;
        let a_end = t_end + params.w_add;
        let m_end = a_end + params.w_mult;
        let sq_end = m_end + params.w_sqrt;
        let sin_end = sq_end + params.w_sin;
        let mod_end = sin_end + params.w_mod;

        if depth == 0 || roll < t_end {
            return Self::terminal(rng);
        }
        if roll < a_end {
            return Expr::Add(
                Box::new(Self::generate(rng, depth - 1, params)),
                Box::new(Self::generate(rng, depth - 1, params)),
            );
        }
        if roll < m_end {
            return Expr::Mult(
                Box::new(Self::generate(rng, depth - 1, params)),
                Box::new(Self::generate(rng, depth - 1, params)),
            );
        }
        if roll < sq_end {
            return Expr::Sqrt(Box::new(Expr::Abs(Box::new(Self::generate(
                rng,
                depth - 1,
                params,
            )))));
        }
        if roll < sin_end {
            return Expr::Sin(Box::new(Self::generate(rng, depth - 1, params)));
        }
        if roll < mod_end {
            return Expr::Mod(
                Box::new(Self::generate(rng, depth - 1, params)),
                Box::new(Self::generate(rng, depth - 1, params)),
            );
        }
        Expr::Mix(
            Box::new(Self::generate(rng, depth - 1, params)),
            Box::new(Self::generate(rng, depth - 1, params)),
            Box::new(Self::generate(rng, depth - 1, params)),
        )
    }

    fn terminal(rng: &mut StdRng) -> Self {
        match rng.r#gen::<u32>() % 7 {
            0 => Expr::Num(rng.gen_range(-1.0f32..=1.0)),
            1 => Expr::X,
            2 => Expr::Y,
            3 => Expr::Abs(Box::new(Expr::X)),
            4 => Expr::Abs(Box::new(Expr::Y)),
            5 => Expr::Sqrt(Box::new(Expr::Add(
                Box::new(Expr::Mult(Box::new(Expr::X), Box::new(Expr::X))),
                Box::new(Expr::Mult(Box::new(Expr::Y), Box::new(Expr::Y))),
            ))),
            6 => Expr::T,
            _ => unreachable!(),
        }
    }
}

fn channel(v: f32) -> u8 {
    (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0) as u8
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::X => write!(f, "x"),
            Expr::Y => write!(f, "y"),
            Expr::T => write!(f, "t"),
            Expr::Num(n) => write!(f, "{n:.3}"),
            Expr::Abs(e) => write!(f, "|{e}|"),
            Expr::Sqrt(e) => write!(f, "√({e})"),
            Expr::Sin(e) => write!(f, "sin(π·{e})"),
            Expr::Add(a, b) => write!(f, "({a} + {b})"),
            Expr::Mult(a, b) => write!(f, "({a} × {b})"),
            Expr::Mod(a, b) => write!(f, "({a} mod {b})"),
            Expr::Mix(a, b, c) => write!(f, "mix({a}, {b}; t={c})"),
        }
    }
}
