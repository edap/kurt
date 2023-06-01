// Vertex shader

struct Camera {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
};
//[[group(1), binding(0)]]
@group(1) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
};
struct InstanceInput {
    // model matrix
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    // normal matrix
    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2
    );
    var res: VertexOutput;
    res.tex_coords = model.tex_coords;
    res.world_normal = normal_matrix * model.normal;
    var world_position: vec4<f32> = model_matrix * vec4<f32>(model.position, 1.0);
    res.world_position = world_position.xyz;
    res.clip_position = camera.view_proj * world_position;
    //res.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    return res;
}



// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

struct Light {
    position: vec3<f32>,
    _pad1: f32,
    color: vec3<f32>,
    _pad2: f32,
};
@group(2) @binding(0)
var<uniform> light: Light;

@fragment
fn fs_main(vo: VertexOutput) -> @location(0) vec4<f32> {
    let object_color = textureSample(t_diffuse, s_diffuse, vo.tex_coords);
    let ambient_strength = 1.0;
    let ambient_color = ambient_strength * light.color;

    // diffuse
    let light_dir = normalize(light.position - vo.world_position);
    let diffuse_strength = max(dot(vo.world_normal, light_dir), 0.0);
    let diffuse_color = light.color * diffuse_strength;

    let result = (ambient_color + diffuse_color) * object_color.xyz;
    return vec4<f32>(result, object_color.a);
}