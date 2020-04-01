use winit::{
    event_loop::{EventLoop, ControlFlow},
    event::*,
    window::Window,
    dpi::PhysicalSize,
};
use std::time::{Instant, Duration};
use image::GenericImageView;
use std::mem::size_of;
use std::path::Path;

fn main() {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();

    let time = Instant::now() + Duration::from_millis(40);

    let id = window.id();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == id => if true {
                *control_flow = ControlFlow::WaitUntil(time);
            } else {
                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::KeyboardInput {
                        input: i,
                        ..
                    } => match i {
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        _ => *control_flow = ControlFlow::WaitUntil(time),
                    },
                    WindowEvent::Resized(size) => {
                        *control_flow = ControlFlow::WaitUntil(time);
                    },
                    WindowEvent::ScaleFactorChanged {
                        new_inner_size,
                        ..
                    } => {
                        *control_flow = ControlFlow::WaitUntil(time);
                    },
                    _ => *control_flow = ControlFlow::WaitUntil(time),
                }
            }
            Event::MainEventsCleared => {
                *control_flow = ControlFlow::WaitUntil(time);
            },
            _ => *control_flow = ControlFlow::WaitUntil(time),
        }
    });
}

#[derive(Debug)]
struct State {
	device: wgpu::Device,
	swap_chain: wgpu::SwapChain,
	queue: wgpu::Queue,
}

impl State {
	fn new(window: &Window) -> Self {
		let instant = Instant::now();
		let inner_size = window.inner_size();
		let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions{
			..Default::default()
		}).unwrap();
		let surface = wgpu::Surface::create(window);

		let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: Default::default(),
        });

        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: inner_size.width,
            height: inner_size.height,
            present_mode: wgpu::PresentMode::Vsync,
        };

        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        Self {
        	device,
        	swap_chain,
        	queue,
        }
	}
}

#[derive(Debug)]
struct Texture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

impl Texture {
    fn from_bytes(device: &wgpu::Device, bytes: &[u8]) -> Result<(Self, wgpu::CommandBuffer), failure::Error> {
        let img = image::load_from_memory(bytes)?;
        Self::from_image(device, &img)
    }

    fn from_image(device: &wgpu::Device, img: &image::DynamicImage) -> Result<(Self, wgpu::CommandBuffer), failure::Error> {
        let rgba = img.as_rgba8().unwrap();
        let dimensions = img.dimensions();

        let size = wgpu::Extent3d{
            width: dimensions.0,
            height: dimensions.1,
            depth: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });

        let buffer = device.create_buffer_mapped(rgba.len(), wgpu::BufferUsage::COPY_SRC).fill_from_slice(&rgba);

        let mut encoder = device.create_command_encoder(&Default::default());

        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &buffer,
                offset: 0,
                row_pitch: 4 * dimensions.0,
                image_height: dimensions.1,
            },
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            size,
        );

        let cmd_buffer = encoder.finish();

        let view = texture.create_default_view();
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare_function: wgpu::CompareFunction::Always,
        });

        Ok((Self { texture, view, sampler }, cmd_buffer))
    }

    fn load<P: AsRef<Path>>(device: &wgpu::Device, path: P) -> Result<(Self, wgpu::CommandBuffer), failure::Error> {
        let img = image::open(path)?;
        Self::from_image(device, &img)
    }
}

trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a>;
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct ModelVertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
}

impl Vertex for ModelVertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        wgpu::VertexBufferDescriptor {
            stride: size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float2,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float3,
                },
            ],
        }
    }
}

#[derive(Debug)]
struct Model {
    meshes: Vec<Mesh>,
    materials: Vec<Material>,
}

#[derive(Debug)]
struct Mesh {
    name: String,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_elements: u32,
    material: usize,
}

#[derive(Debug)]
struct Material {
    name: String,
    diffuse_texture: Texture,
}

impl Model {
    fn load<P: AsRef<Path>>(device: &wgpu::Device, path: P) -> Result<(Self, Vec<wgpu::CommandBuffer>), failure::Error> {
        let (obj_models, obj_materials) = tobj::load_obj(path.as_ref())?;

        let folder = path.as_ref().parent().unwrap();

        let mut command_buffers = Vec::new();

        let mut materials = Vec::new();
        for mat in obj_materials {
            let diffuse_path = mat.diffuse_texture;
            let (diffuse_texture, cmds) = Texture::load(&device, folder.join(diffuse_path))?;
            materials.push(Material {
                name: mat.name,
                diffuse_texture,
            });
            command_buffers.push(cmds);
        }

        let mut meshes = Vec::new();
        for m in obj_models {
            let mut vertices = Vec::new();
            for i in 0..m.mesh.position.len() / 3 {
                vertices.push(ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [
                        m.mesh.texcoords[i * 2],
                        m.mesh.texcoords[i * 2 + 1],
                    ],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                });
            }

            let vertex_buffer = device.create_buffer_mapped(vertices.len(), wgpu::BufferUsage::VERTEX).fill_from_slice(&vertices);
            let index_buffer = device.create_buffer_mapped(m.mesh.indices.len(), wgpu::BufferUsage::INDEX).fill_from_slice(&m.mesh.indices);

            meshed.push(Mesh {
                name: m.name,
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            });
        }

        Ok((Self { meshes, materials, }, command_buffers))
    }
}