#version 460 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 WorldPos;
in vec3 Normal;
in mat3 TBN;

// Material parameters
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicRoughnessMap;
uniform sampler2D emissiveMap;
uniform sampler2D aoMap;

// Flags to check if textures exist (optional, but good for robustness)
uniform bool hasAlbedoMap;
uniform bool hasNormalMap;
uniform bool hasMetallicRoughnessMap;
uniform bool hasEmissiveMap;
uniform bool hasAoMap;

// Fallback values
uniform vec3 uAlbedo;
uniform float uMetallic;
uniform float uRoughness;

// Lights
uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 camPos;

const float PI = 3.14159265359;

// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / max(denom, 0.0000001); // prevent divide by zero
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
// ----------------------------------------------------------------------------
void main()
{		
    // Sample textures
    vec3 albedo = hasAlbedoMap ? pow(texture(albedoMap, TexCoords).rgb, vec3(2.2)) : uAlbedo;
    float metallic = hasMetallicRoughnessMap ? texture(metallicRoughnessMap, TexCoords).b : uMetallic;
    float roughness = hasMetallicRoughnessMap ? texture(metallicRoughnessMap, TexCoords).g : uRoughness;
    float ao = hasAoMap ? texture(aoMap, TexCoords).r : 1.0;
    vec3 emissive = hasEmissiveMap ? texture(emissiveMap, TexCoords).rgb : vec3(0.0);
    
    // Normal mapping
    vec3 N = normalize(Normal);
    if (hasNormalMap) {
        vec3 normalSample = texture(normalMap, TexCoords).rgb;
        normalSample = normalSample * 2.0 - 1.0;   
        N = normalize(TBN * normalSample); 
    }

    vec3 V = normalize(camPos - WorldPos);

    // Calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    // Reflectance equation
    vec3 Lo = vec3(0.0);

    // Calculate per-light radiance (Directional Light)
    vec3 L = normalize(-lightDir); // Direction FROM surface TO light
    vec3 H = normalize(V + L);
    
    // Directional light has no attenuation
    vec3 radiance = lightColor;

    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);   
    float G   = GeometrySmith(N, V, L, roughness);      
    vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
       
    vec3 numerator    = NDF * G * F; 
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
    vec3 specular = numerator / denominator;
    
    // kS is equal to Fresnel
    vec3 kS = F;
    // for energy conservation, the diffuse and specular light can't
    // be above 1.0 (unless the surface emits light); to preserve this
    // relationship the diffuse component (kD) should equal 1.0 - kS.
    vec3 kD = vec3(1.0) - kS;
    // multiply kD by the inverse metalness such that only non-metals 
    // have diffuse lighting, or a linear blend if partly metal (pure metals
    // have no diffuse light).
    kD *= 1.0 - metallic;	  

    // Scale light by NdotL
    float NdotL = max(dot(N, L), 0.0);        

    // add to outgoing radiance Lo
    Lo += (kD * albedo / PI + specular) * radiance * NdotL;  
    
    // Ambient lighting (simple constant ambient)
    // In a real PBR renderer, we'd use IBL (Image Based Lighting)
    // Here we use a simple ambient term + AO
    vec3 ambient = vec3(0.03) * albedo * ao;
    
    vec3 color = ambient + Lo + emissive;

    // HDR tonemapping (Reinhard)
    color = color / (color + vec3(1.0));
    // Gamma correction
    color = pow(color, vec3(1.0/2.2)); 

    FragColor = vec4(color, 1.0);
}
