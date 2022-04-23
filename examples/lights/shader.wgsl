// Vertex shader

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
};
@group(2) @binding(0)
var<uniform> light: Light;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    // We need to reassemble the matrix before we can use it.
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    var res: VertexOutput;
    res.tex_coords = model.tex_coords;
    // res.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    res.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    // we do apply the model_matrix before we apply the camera.view_proj (remember, matrices are applied
    // in the inverse sense from what you read, the camera.view_proj is the last matrix to be
    // applied). This is because the camera.view_proj change the coordinate system from local to
    // world. Our model matrix is a world space transformation, we do not want it in camera space.
    return res;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;


@fragment
fn fs_main(vo: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, vo.tex_coords);
}