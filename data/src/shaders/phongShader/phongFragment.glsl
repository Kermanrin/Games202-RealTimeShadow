#ifdef GL_ES
precision mediump float;
#endif

// #define LIGHT1
// #define LIGHT2

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;


// Shadow map related variables
#define NUM_SAMPLES 100
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

uniform sampler2D uShadowMap;
uniform sampler2D uShadowMap1;
uniform sampler2D uShadowMap2;

varying vec4 vPositionFromLight;
varying vec4 vPositionFromLight1;
varying vec4 vPositionFromLight2;
float offset;

float _Offset() {
  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float offset = 1.0 / (2048.0 * max(0.1, abs(dot(normal, lightDir))));//Shadow Map的偏移量
  return abs(offset);
}

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}

float unpack(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0, 1.0/256.0, 1.0/(256.0*256.0), 1.0/(256.0*256.0*256.0));
    return dot(rgbaDepth, bitShift);

}

vec2 poissonDisk[NUM_SAMPLES];

void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.5 );
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}

void uniformDiskSamples( const in vec2 randomSeed ) {//均匀采样

  float randNum = rand_2to1(randomSeed);//生成一个二维的[0.1]随机变量
  float sampleX = rand_1to1( randNum ) ;//生成一个一维的[-1,1]得随机变量
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);//由密度变换公式，以sqrt(x)为半径满足均匀分布

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );//将生成的偏移量存储
    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;
    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

float findBlocker( sampler2D shadowMap,  vec2 uv, float zReceiver ) {//返回平局block深度
	 
  uniformDiskSamples(uv);
  // poissonDiskSamples(uv);
  float textureSize = 400.0;

  // 注意 block 的步长要比 PCSS 中的 PCF 步长长一些，这样生成的软阴影会更加柔和
  float filterStride = 25.0;
  float filterRange = 1.0 / textureSize * filterStride;

  // 有多少点在阴影里
  int shadowCount = 0;
  float blockDepth = 0.0;
  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    vec2 sampleCoord = poissonDisk[i] * filterRange + uv;
    vec4 closestDepthVec = texture2D(shadowMap, sampleCoord); 
    float closestDepth = unpack(closestDepthVec);
    if(zReceiver > closestDepth + offset){
      blockDepth += closestDepth;
      shadowCount += 1;
    }
}

  if(shadowCount==NUM_SAMPLES){
    return 2.0;
  }

	return blockDepth / float(shadowCount);
}

float PCF(sampler2D shadowMap, vec4 coords) {
  // poissonDiskSamples(coords.xy);//泊松
  uniformDiskSamples(coords.xy);//均匀

  // shadow map 的大小, 越大滤波的范围越小
  float textureSize = 400.0;
  // 滤波的步长
  float filterStride = 5.0;
  // 滤波窗口的范围
  float filterRange = 1.0 / textureSize * filterStride;
  // 有多少点不在阴影里
  int noShadowCount = 0;
  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    vec2 sampleCoord = poissonDisk[i] * filterRange + coords.xy;
    vec4 closestDepthVec = texture2D(shadowMap, sampleCoord); 
    float closestDepth = unpack(closestDepthVec);//平面降噪
    if (abs(closestDepth) < EPS) closestDepth = 1.0;
    float currentDepth = coords.z;
    if(currentDepth < closestDepth + 0.01){
      noShadowCount += 1;
    }
  }

  float shadow = float(noShadowCount) / float(NUM_SAMPLES);//计算均值
  return shadow;
}

float PCSS(sampler2D shadowMap, vec4 coords){

  float zReceiver = coords.z;

  // STEP 1: avgblocker depth
  float zBlocker = findBlocker(shadowMap, coords.xy, zReceiver);
  if(zBlocker < EPS) return 1.0;
  if(zBlocker > 1.0) return 0.0;

  // STEP 2: penumbra size
  float wPenumbra = (zReceiver - zBlocker) / zBlocker;

  // STEP 3: filtering
  float textureSize = 400.0;
  // 这里的步长要比 STEP 1 的步长小一些
  float filterStride = 5.0;
  float filterRange = 1.0 / textureSize * filterStride * wPenumbra;
  int noShadowCount = 0;
  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    vec2 sampleCoord = poissonDisk[i] * filterRange + coords.xy;
    vec4 closestDepthVec = texture2D(shadowMap, sampleCoord); 
    float closestDepth = unpack(closestDepthVec);//平面降噪
    if (abs(closestDepth) < EPS) closestDepth = 1.0;
    float currentDepth = coords.z;
    if(currentDepth < closestDepth + 0.01){
      noShadowCount += 1;
    }
  }

  float shadow = float(noShadowCount) / float(NUM_SAMPLES);
  return shadow;
}


float useShadowMap(sampler2D shadowMap, vec4 shadowCoord){
  // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
  
  float Depth_of_light = unpack(texture2D(shadowMap, shadowCoord.xy));
  // get depth of current fragment from light's perspective
  if (abs(Depth_of_light) < EPS) Depth_of_light = 1.0;//平面降噪
  float Depth_of_camera = shadowCoord.z;
  // check whether current frag pos is in shadow
  if (Depth_of_camera < Depth_of_light+offset)
  {
    return 1.0;
    }
  else 
  {
    return 0.0;
    }
}

vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  vec3 ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}

void main(void) {

  offset = _Offset();//偏移量
  vec3 shadowCoord = vPositionFromLight.xyz  ;
  // 归一化至 [0,1] 
  shadowCoord = shadowCoord * 0.5 + 0.5;
  float visibility = 1.0 , visibility2 = 1.0 , visibility1=1.0;

  // visibility = useShadowMap(uShadowMap, vec4(shadowCoord, 1.0));//shadowmap算法
  // visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0));//PCF算法
  visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0));//PCSS算法


  // //光源二
  #if defined(LIGHT1)
  vec3 shadowCoord1 = vPositionFromLight1.xyz  ;
  shadowCoord1 = shadowCoord1 * 0.5 + 0.5;
  visibility1 = useShadowMap(uShadowMap1, vec4(shadowCoord1 , 1.0));
  // visibility1 = PCF(uShadowMap1, vec4(shadowCoord1, 1.0));
  // visibility1 = PCSS(uShadowMap1, vec4(shadowCoord1, 1.0));

  visibility = (visibility + visibility1) / 2.0;
  #endif


  // 光源三
  #if defined(LIGHT2)
  vec3 shadowCoord2 = vPositionFromLight2.xyz  ;
  shadowCoord2 = shadowCoord2 * 0.5 + 0.5;
  visibility2 = useShadowMap(uShadowMap2, vec4(shadowCoord2, 1.0));
  // visibility2 = PCF(uShadowMap2, vec4(shadowCoord2, 1.0));
  // visibility2 = PCSS(uShadowMap2, vec4(shadowCoord2, 1.0));

  visibility = (visibility + visibility2) / 2.0;
  #endif

 

  vec3 phongColor = blinnPhong();

  gl_FragColor = vec4(phongColor * visibility, 1.0);
  // gl_FragColor = vec4(phongColor, 1.0);
}