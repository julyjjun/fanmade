import os
import base64
import json
import requests
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from PIL import Image
import io

# ================================
# 🔧 설정 부분 (여기를 수정하세요)
# ================================

# Google Cloud 설정
GOOGLE_CLOUD_PROJECT_ID = "aiapi-468707"  # 실제 프로젝트 ID
GOOGLE_CLOUD_LOCATION = "us-central1"
GOOGLE_APPLICATION_CREDENTIALS = "C:/Users/ssu/aiAPI/aiAPI/credentials/aiapi-468707-40156040e383.json"  # 서비스 계정 키 파일 경로

# 🎨 변수 기반 프롬프트 템플릿
PROMPT_TEMPLATE = """
K-POP comback celebration advertisement background design,
{background_color} {mood} background with {effect} effects,
clear empty areas on both the left and right sides reserved for text placement,
no actual text, instead bright blurred zones to indicate text placement,
strictly no text anywhere in the image,
no people, no characters, background only.
"""

# 📋 변수 기본값들
DEFAULT_VARIABLES = {                   
    "background_color": "dark and mysterious black-to-purple gradient",
    "mood": "mysteric",
    "effect": "sparkling"
}

# 📊 이미지 생성 파라미터
IMAGE_PARAMS = {
    "number_of_images": 2,           # 생성할 이미지 개수 (1-8)
    "aspect_ratio": "16:9",          # 화면 비율 (1:1, 9:16, 16:9, 4:3, 3:4)
    "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy, text, watermark",  # 제외할 요소
    "person_generation": "allow_all", # 인물 생성 (allow_all, allow_adult, block_all)
    "safety_filter_level": "block_few", # 안전 필터 (block_few, block_some, block_most)
    "add_watermark": False,          # 워터마크 추가 여부
    "guidance_scale": 7,             # 프롬프트 가이던스 강도 (1-20)
    "seed": None                     # 시드값 (None이면 랜덤)
}

# 📁 출력 폴더
OUTPUT_DIR = "generated_images"

# ================================
# 🔧 새로운 변수 처리 함수들
# ================================

def get_user_input():
    """사용자로부터 변수 입력받기"""
    print("\n🎤 광고 이미지 설정을 입력해주세요 (Enter로 기본값 사용)")
    print("=" * 50)
    
    variables = {}

    # 배경 스타일 입력
    background_color = input(f"🖼️ 배경 스타일 [{DEFAULT_VARIABLES['background_color']}]: ").strip()
    if background_color:
        variables['background_color'] = background_color
    else:
        variables['background_color'] = DEFAULT_VARIABLES['background_color']

    # 분위기 입력
    mood = input(f"🖼️ 분위기 [{DEFAULT_VARIABLES['mood']}]: ").strip()
    if mood:
        variables['mood'] = mood
    else:
        variables['mood'] = DEFAULT_VARIABLES['mood']

    # 효과 입력
    effect = input(f"🖼️ 효과 [{DEFAULT_VARIABLES['effect']}]: ").strip()
    if effect:
        variables['effect'] = effect
    else:
        variables['effect'] = DEFAULT_VARIABLES['effect']

    return variables

def create_prompt_from_template(variables):
    """템플릿과 변수로 프롬프트 생성"""
    try:
        prompt = PROMPT_TEMPLATE.format(**variables)
        return prompt.strip()
    except KeyError as e:
        print(f"❌ 변수 오류: {e}")
        return None

def print_generated_prompt(prompt, variables):
    """생성된 프롬프트와 변수 출력"""
    print("\n📝 적용된 변수:")
    print("-" * 30)
    for key, value in variables.items():
        print(f"  {key}: {value}")
    
    print(f"\n🎨 생성된 프롬프트:")
    print("-" * 30)
    print(prompt)
    print("-" * 30)


def interactive_mode():
    """대화형 모드"""
    print("🎯 입력 방식을 선택해주세요:")
    print("1. 직접 입력")
    
    choice = input("선택 (1): ").strip()
    
    return get_user_input()

    

# ================================
# 🔧 기존 함수들 (수정 없음)
# ================================

def setup_directories():
    """출력 디렉토리 생성"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ 출력 폴더 준비: {OUTPUT_DIR}")

def get_access_token():
    """Google Cloud 액세스 토큰 획득"""
    try:
        if not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
            raise FileNotFoundError(f"인증 파일을 찾을 수 없습니다: {GOOGLE_APPLICATION_CREDENTIALS}")
        
        # 서비스 계정으로 인증
        credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_APPLICATION_CREDENTIALS,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        credentials.refresh(Request())
        
        print("✅ Google Cloud 인증 성공")
        return credentials.token
        
    except Exception as e:
        print(f"❌ 인증 실패: {e}")
        return None

def generate_images(prompt, params):
    """Google Cloud Imagen으로 이미지 생성"""
    try:
        # 액세스 토큰 획득
        access_token = get_access_token()
        if not access_token:
            return False
        
        # 사용 가능한 모델들
        models_to_try = [
            "imagen-3.0-generate-002"
        ]
        
        for model in models_to_try:
            try:
                print(f"🎨 {model} 모델로 이미지 생성 시도 중...")
                
                # API 엔드포인트
                url = f"https://{GOOGLE_CLOUD_LOCATION}-aiplatform.googleapis.com/v1/projects/{GOOGLE_CLOUD_PROJECT_ID}/locations/{GOOGLE_CLOUD_LOCATION}/publishers/google/models/{model}:predict"
                
                # 헤더
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }
                
                # 요청 데이터 구성
                if "imagegeneration" in model:
                    # Vertex AI Image Generation 모델용
                    request_data = {
                        "instances": [
                            {
                                "prompt": prompt.strip()
                            }
                        ],
                        "parameters": {
                            "sampleCount": params["number_of_images"],
                            "language": "en",
                            "aspectRatio": params["aspect_ratio"],
                            "safetyFilterLevel": params["safety_filter_level"],
                            "personGeneration": params["person_generation"],
                            "addWatermark": params["add_watermark"]
                        }
                    }
                    
                    # 네거티브 프롬프트가 있으면 추가
                    if params["negative_prompt"]:
                        request_data["parameters"]["negativePrompt"] = params["negative_prompt"]
                        
                    # 시드가 있으면 추가
                    if params["seed"]:
                        request_data["parameters"]["seed"] = params["seed"]
                        
                else:
                    # Imagen 3.0 모델용
                    request_data = {
                        "instances": [
                            {
                                "prompt": prompt.strip(),
                                "sampleCount": params["number_of_images"]
                            }
                        ],
                        "parameters": {
                            "aspectRatio": params["aspect_ratio"],
                            "safetyFilterLevel": params["safety_filter_level"],
                            "personGeneration": params["person_generation"],
                            "addWatermark": params["add_watermark"],
                            "guidanceScale": params["guidance_scale"]
                        }
                    }
                
                print(f"📤 API 호출 중...")
                print(f"📝 요청 파라미터: {json.dumps(request_data['parameters'], indent=2)}")
                
                # API 호출
                response = requests.post(url, headers=headers, json=request_data, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ {model} 호출 성공!")
                    
                    # 응답에서 이미지 저장
                    if save_images_from_response(result, prompt):
                        return True
                    
                else:
                    error_detail = response.json() if response.content else {}
                    print(f"❌ {model} 실패 (HTTP {response.status_code}): {error_detail}")
                    continue
                    
            except requests.exceptions.Timeout:
                print(f"⏰ {model} 시간 초과, 다음 모델 시도...")
                continue
            except Exception as model_error:
                print(f"❌ {model} 오류: {model_error}")
                continue
        
        print("❌ 모든 모델에서 이미지 생성 실패")
        return False
        
    except Exception as e:
        print(f"❌ 이미지 생성 실패: {e}")
        return False

def save_images_from_response(result, prompt):
    """API 응답에서 이미지를 추출하여 저장"""
    try:
        if 'predictions' not in result or not result['predictions']:
            print("❌ 응답에 예측 결과가 없음")
            return False
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 각 예측 결과 처리
        saved_count = 0
        for i, prediction in enumerate(result['predictions']):
            
            # 이미지 데이터 추출 (다양한 구조 지원)
            image_data = None
            possible_paths = [
                'bytesBase64Encoded',
                'generatedImage.bytesBase64Encoded',
                'image.bytesBase64Encoded',
                'images[0].bytesBase64Encoded'
            ]
            
            for path in possible_paths:
                try:
                    if '.' in path:
                        # 중첩된 객체 접근
                        keys = path.split('.')
                        value = prediction
                        for key in keys:
                            if '[' in key and ']' in key:
                                # 배열 인덱스 처리
                                array_key = key.split('[')[0]
                                index = int(key.split('[')[1].split(']')[0])
                                value = value[array_key][index]
                            else:
                                value = value[key]
                        image_data = value
                    else:
                        image_data = prediction.get(path)
                    
                    if image_data:
                        break
                        
                except (KeyError, IndexError, TypeError):
                    continue
            
            if image_data:
                try:
                    # Base64 디코딩
                    image_bytes = base64.b64decode(image_data)
                    
                    # PIL Image로 변환
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # 파일명 생성
                    filename = f"generated_{timestamp}_{i+1:02d}.png"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    
                    # 이미지 저장
                    image.save(filepath, "PNG", quality=95)
                    
                    print(f"💾 이미지 저장: {filename} ({image.size[0]}x{image.size[1]})")
                    saved_count += 1
                    
                except Exception as save_error:
                    print(f"❌ 이미지 {i+1} 저장 실패: {save_error}")
                    continue
            else:
                print(f"⚠️ 예측 {i+1}에서 이미지 데이터를 찾을 수 없음")
                if i == 0:  # 첫 번째 예측에서 구조 출력
                    print(f"🔍 응답 구조: {list(prediction.keys()) if isinstance(prediction, dict) else type(prediction)}")
        
        if saved_count > 0:
            print(f"✅ 총 {saved_count}개 이미지 저장 완료!")
            return True
        else:
            print("❌ 저장된 이미지가 없음")
            return False
            
    except Exception as e:
        print(f"❌ 이미지 저장 처리 실패: {e}")
        return False

def print_settings(variables):
    """현재 설정 출력"""
    print("=" * 60)
    print("🎨 Google Cloud Imagen 이미지 생성기 (변수 버전)")
    print("=" * 60)
    print("📋 적용된 변수:")
    for key, value in variables.items():
        print(f"   {key}: {value}")
    print(f"📊 생성 개수: {IMAGE_PARAMS['number_of_images']}")
    print(f"📐 화면 비율: {IMAGE_PARAMS['aspect_ratio']}")
    print(f"🔒 안전 필터: {IMAGE_PARAMS['safety_filter_level']}")
    print(f"👤 인물 생성: {IMAGE_PARAMS['person_generation']}")
    print(f"💧 워터마크: {IMAGE_PARAMS['add_watermark']}")
    print(f"📁 출력 폴더: {OUTPUT_DIR}")
    print("=" * 60)

def main():
    """메인 실행 함수"""
    try:
        print("🎵 K-POP 아이돌 그룹 이미지 생성기")
        
        # 사용자로부터 변수 입력받기
        variables = interactive_mode()
        
        # 프롬프트 생성
        prompt = create_prompt_from_template(variables)
        if not prompt:
            print("❌ 프롬프트 생성 실패")
            return
        
        # 설정 출력
        print_settings(variables)
        
        # 생성된 프롬프트 출력
        print_generated_prompt(prompt, variables)
        
        # 계속 진행할지 확인
        continue_choice = input(f"\n🚀 이 설정으로 이미지를 생성하시겠습니까? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes', '']:
            print("⏹️ 생성 취소됨")
            return
        
        # 디렉토리 준비
        setup_directories()
        
        # 필수 설정 확인
        if not GOOGLE_CLOUD_PROJECT_ID:
            print("❌ GOOGLE_CLOUD_PROJECT_ID가 설정되지 않았습니다")
            return
        
        if not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
            print(f"❌ 서비스 계정 키 파일이 없습니다: {GOOGLE_APPLICATION_CREDENTIALS}")
            print("💡 Google Cloud Console에서 서비스 계정 키를 다운로드하고 경로를 수정해주세요")
            return
        
        # 이미지 생성 시작
        print("\n🚀 이미지 생성 시작...")
        success = generate_images(prompt, IMAGE_PARAMS)
        
        if success:
            print(f"\n🎉 완료! {OUTPUT_DIR} 폴더를 확인해보세요")
        else:
            print("\n😞 이미지 생성에 실패했습니다")
            
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {e}")

if __name__ == "__main__":
    main()