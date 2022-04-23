

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

// Vertex shader
@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var res: VertexOutput;
    res.color = model.color;
    res.clip_position = vec4<f32>(model.position, 1.0);
    return res;
}

// Fragment shader
@fragment
fn fs_main(vo: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(vo.color, 1.0);
}