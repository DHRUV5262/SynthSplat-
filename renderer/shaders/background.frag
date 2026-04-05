#version 460 core
out vec4 FragColor;

in vec3 WorldPos;

void main()
{
    // Simple solid light grey background - less confusing for 3DGS
    vec3 color = vec3(0.7, 0.7, 0.75); // Light grey with slight blue tint
    
    // Apply gamma correction to match PBR output
    color = pow(color, vec3(1.0/2.2));
    
    FragColor = vec4(color, 1.0);
}
