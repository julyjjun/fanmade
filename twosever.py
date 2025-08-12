from fastapi import FastAPI, Form ,HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import aiplatform
from google.protobuf import struct_pb2
from google.protobuf.struct_pb2 import Value
import asyncio
import requests
from rembg import remove
from PIL import Image, ImageDraw, ImageFont, ImageColor,ImageEnhance, ImageFilter
from bs4 import BeautifulSoup
import io
import base64
import random
import time
import math
import json
import openai
import os
import re
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import re
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator
import urllib.parse
from openai import OpenAI

# 기존 FastAPI 설정은 그대로...
app = FastAPI(title="Fan Ad AI Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 1. 환경변수 로드 (.env + OS 환경)
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


# Vertex AI 초기화
if GOOGLE_CLOUD_PROJECT_ID:
    try:
        aiplatform.init(project=GOOGLE_CLOUD_PROJECT_ID, location=GOOGLE_CLOUD_LOCATION)
        print(f"✅ Google Cloud Vertex AI 초기화 완료 (프로젝트: {GOOGLE_CLOUD_PROJECT_ID})")
    except Exception as e:
        print(f"❌ Google Cloud 초기화 실패: {e}")
else:
    print("⚠️ GOOGLE_CLOUD_PROJECT_ID가 설정되지 않았습니다")

async def generate_background_with_google_imagen(prompt):
    """🎨 Google Cloud Imagen API로 고품질 이미지 생성"""
    try:
        print(f"🎨 Google Cloud Imagen으로 이미지 생성 중...")
        print(f"📝 프롬프트: {prompt[:100]}...")
        
        # Imagen 모델 엔드포인트 설정
        model_name = "imagen-3.0-generate-001"  # 최신 Imagen 모델
        endpoint_path = f"projects/{GOOGLE_CLOUD_PROJECT_ID}/locations/{GOOGLE_CLOUD_LOCATION}/publishers/google/models/{model_name}"
        
        # 이미지 생성 파라미터
        instances = [
            {
                "prompt": prompt,
                "sampleCount": 4,
                "aspectRatio": "1:1",  # 1024x1024 기본
                "safetyFilterLevel": "block_some",
                "personGeneration": "allow_adult",
                "addWatermark": False,  # 워터마크 제거
                "seed": None,  # 랜덤 시드
                "guidanceScale": 7,  # 프롬프트 가이던스 (1-20)
                "negativePrompt": "blurry, low quality, distorted, ugly, bad anatomy",
            }
        ]
        
        parameters = {
            "sampleCount": 1,
            "language": "en",  # 영어로 고정
            "safetyFilterLevel": "block_some"
        }
        
        # Vertex AI 클라이언트로 예측 요청
        from google.cloud import aiplatform_v1
        
        client = aiplatform_v1.PredictionServiceClient()
        
        # 요청 데이터 구성
        instances_proto = []
        for instance in instances:
            instance_proto = struct_pb2.Value()
            instance_proto.struct_value.update(instance)
            instances_proto.append(instance_proto)
        
        parameters_proto = struct_pb2.Value()
        parameters_proto.struct_value.update(parameters)
        
        # 예측 요청 실행
        response = client.predict(
            endpoint=endpoint_path,
            instances=instances_proto,
            parameters=parameters_proto
        )
        
        # 응답에서 이미지 데이터 추출
        if response.predictions:
            prediction = response.predictions[0]
            
            # 이미지 데이터 추출 방법 (Imagen 응답 구조에 따라)
            if hasattr(prediction, 'struct_value'):
                image_data = None
                
                # 여러 가능한 필드명 시도
                possible_fields = [
                    'bytesBase64Encoded',
                    'image',
                    'generated_image',
                    'output'
                ]
                
                for field in possible_fields:
                    if field in prediction.struct_value.fields:
                        image_data = prediction.struct_value.fields[field].string_value
                        break
                
                if not image_data:
                    # 전체 응답 구조 확인을 위한 디버깅
                    print("🔍 응답 구조 분석 중...")
                    print(f"Available fields: {list(prediction.struct_value.fields.keys())}")
                    raise Exception("응답에서 이미지 데이터를 찾을 수 없음")
                
                # Base64 디코딩
                try:
                    image_bytes = base64.b64decode(image_data)
                except Exception as decode_error:
                    print(f"Base64 디코딩 실패: {decode_error}")
                    # 직접 바이너리 데이터인 경우
                    image_bytes = image_data.encode() if isinstance(image_data, str) else image_data
                
                # PIL Image로 변환
                background_image = Image.open(io.BytesIO(image_bytes))
                
                # 이미지 품질 확인 및 향상
                width, height = background_image.size
                if width >= 512 and height >= 512:
                    print(f"✅ Google Cloud Imagen 이미지 생성 성공! ({width}x{height})")
                    return background_image
                else:
                    print(f"⚠️ 낮은 해상도 ({width}x{height}), 품질 향상 중...")
                    return enhance_image_quality(background_image)
            else:
                raise Exception("예상과 다른 응답 구조")
        else:
            raise Exception("응답에 이미지가 없음")
            
    except Exception as e:
        print(f"❌ Google Cloud Imagen 실패: {e}")
        print("🔄 고품질 로컬 배경으로 대체...")
        return create_premium_background()

async def generate_background_with_google_imagen_simple(prompt):
    """🎨 간단한 Google Cloud Imagen API 호출 (HTTP REST 방식)"""
    try:
        print(f"🎨 Google Cloud Imagen (REST API)로 이미지 생성 중...")
        
        # 구글 클라우드 액세스 토큰 획득
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
        
        # 서비스 계정 키로 인증
        credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_APPLICATION_CREDENTIALS,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        credentials.refresh(Request())
        access_token = credentials.token
        
        # Imagen API 엔드포인트
        url = f"https://{GOOGLE_CLOUD_LOCATION}-aiplatform.googleapis.com/v1/projects/{GOOGLE_CLOUD_PROJECT_ID}/locations/{GOOGLE_CLOUD_LOCATION}/publishers/google/models/imagen-3.0-generate-001:predict"
        
        # 요청 헤더
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # 요청 데이터
        data = {
            "instances": [
                {
                    "prompt": prompt,
                    "sampleCount": 1,
                    "aspectRatio": "1:1",
                    "safetyFilterLevel": "block_some",
                    "personGeneration": "allow_adult"
                }
            ],
            "parameters": {
                "sampleCount": 1
            }
        }
        
        # API 호출
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # 응답에서 이미지 추출
        if 'predictions' in result and result['predictions']:
            prediction = result['predictions'][0]
            
            # 이미지 데이터 추출
            image_data = None
            if 'bytesBase64Encoded' in prediction:
                image_data = prediction['bytesBase64Encoded']
            elif 'generatedImage' in prediction:
                image_data = prediction['generatedImage']['bytesBase64Encoded']
            
            if image_data:
                # Base64 디코딩 후 이미지 생성
                image_bytes = base64.b64decode(image_data)
                background_image = Image.open(io.BytesIO(image_bytes))
                
                print(f"✅ Google Cloud Imagen (REST) 성공! {background_image.size}")
                return background_image
            else:
                print("🔍 응답 구조:", json.dumps(result, indent=2)[:500])
                raise Exception("응답에서 이미지 데이터를 찾을 수 없음")
        else:
            raise Exception("API 응답에 예측 결과가 없음")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP 요청 실패: {e}")
        return create_premium_background()
    except Exception as e:
        print(f"❌ Google Cloud Imagen (REST) 실패: {e}")
        return create_premium_background()
    
# 광고 옵션 설정
AD_OPTIONS = {
    "background_color": {
        "blue": "#4A90E2",
        "pink": "#FF6B9D", 
        "white": "#FFFFFF",
        "gold": "#FFD700",
        "purple": "#9B59B6"
    },
    "mood": {
        "bright_fresh": "밝고 청량한",
        "romantic": "로맨틱한",
        "luxury": "고급스러운", 
        "energetic": "활기찬",
        "cute": "귀여운"
    },
    "effects": {
        "sparkle": "반짝이는 효과",
        "gradient": "그라데이션 효과",
        "pattern": "패턴 장식",
        "simple": "심플한 스타일",
        "neon": "네온 효과"
    }
}


# 결과 이미지를 저장할 폴더 만들기
os.makedirs("generated_ads", exist_ok=True)
            
def generate_prompt(background_color, mood, effect, width, height):
    """🎨 유연한 색상 처리가 포함된 프롬프트 생성 함수"""
    

    # 기본 효과
    effects_simple = {
        "sparkle": "with sparkle effects",
        "gradient": "with gradient effects",
        "pattern": "with pattern effects", 
        "simple": "clean and simple",
        "neon": "with neon glow effects"
    }
    
    effect = effects_simple.get(effect, "")     

    
    # 극단적으로 단순한 프롬프트
    optimized_prompt = f"""
A {effect} {background_color} background with a {mood} feeling
Size is {width}x{height} pixels
"""
    

    # 🔍 프롬프트 저장 (디버깅용)
    request_info = {
        "background_color": background_color,
        "mood": mood,
        "effects": effect
    }
    
    save_prompt_to_history(optimized_prompt.strip(), request_info)
    
    return optimized_prompt.strip()

# 프롬프트 저장용 전역 변수
PROMPT_HISTORY = []

def save_prompt_to_history(prompt, request_info):
    """생성된 프롬프트를 히스토리에 저장"""
    
    prompt_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "request_info": request_info,
        "prompt": prompt,
        "prompt_length": len(prompt),
        "id": len(PROMPT_HISTORY) + 1
    }
    
    PROMPT_HISTORY.append(prompt_entry)
    
    # 최근 50개만 유지 (메모리 절약)
    if len(PROMPT_HISTORY) > 50:
        PROMPT_HISTORY.pop(0)
    
    print(f"📝 프롬프트 저장됨 (ID: {prompt_entry['id']})")

def remove_background_from_image(image_file):
    """업로드된 이미지에서 배경 제거하는 함수"""
    # 업로드된 파일을 바이트로 읽기
    image_bytes = image_file.read()
    
    # rembg로 배경 제거
    output_bytes = remove(image_bytes)
    
    # PIL Image로 변환
    no_bg_image = Image.open(io.BytesIO(output_bytes))
    
    return no_bg_image

async def generate_background_with_ai(prompt):
    """🚀 메인 이미지 생성 함수 - Google Cloud Imagen 사용"""
    
    # Google Cloud 설정 확인
    if not all([GOOGLE_CLOUD_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS]):
        print("⚠️ Google Cloud 설정이 불완전합니다. 설정을 확인해주세요.")
        print(f"프로젝트 ID: {'✅' if GOOGLE_CLOUD_PROJECT_ID else '❌'}")
        print(f"인증 파일: {'✅' if GOOGLE_APPLICATION_CREDENTIALS else '❌'}")
        return create_premium_background()
    
    # 인증 파일 존재 확인
    if not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        print(f"❌ 인증 파일을 찾을 수 없습니다: {GOOGLE_APPLICATION_CREDENTIALS}")
        return create_premium_background()
    
    # Google Cloud Imagen 시도 (REST API 방식 우선)
    try:
        result = await generate_background_with_google_imagen_simple(prompt)
        if result:
            return result
    except Exception as e:
        print(f"REST API 방식 실패: {e}")
    
    # 클라이언트 라이브러리 방식 시도
    try:
        result = await generate_background_with_google_imagen(prompt)
        if result:
            return result
    except Exception as e:
        print(f"클라이언트 라이브러리 방식 실패: {e}")
    
    # 모든 방식 실패시 로컬 배경 생성
    print("🔄 Google Cloud 호출 실패, 고품질 로컬 배경 생성 중...")
    return create_premium_background()

        

        
def create_premium_background():
    """🎨 AI 실패시 사용할 고품질 그라데이션 배경"""
    from PIL import ImageDraw
    
    # 고품질 그라데이션 배경 생성
    size = (1024, 1024)
    background = Image.new('RGB', size)
    draw = ImageDraw.Draw(background)
    
    # 아름다운 그라데이션 생성
    for y in range(size[1]):
        # 핑크에서 퍼플로 그라데이션
        ratio = y / size[1]
        r = int(255 * (1 - ratio) + 147 * ratio)  # 255 -> 147
        g = int(107 * (1 - ratio) + 112 * ratio)  # 107 -> 112  
        b = int(157 * (1 - ratio) + 219 * ratio)  # 157 -> 219
        
        color = (r, g, b)
        draw.line([(0, y), (size[0], y)], fill=color)
    
    print("✅ 프리미엄 그라데이션 배경 생성 완료")
    return background

def enhance_image_quality(image):
    """🔧 이미지 품질 향상 필터"""
    from PIL import ImageEnhance, ImageFilter
    
    # 1. 선명도 증가
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3)
    
    # 2. 색상 채도 향상
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.2)
    
    # 3. 대비 개선
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    
    # 4. 밝기 최적화
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.05)
    
    print("✅ 이미지 품질 향상 완료")
    return image

def add_text_to_image(image, text, person_position="top"):
    """🎨 프리미엄 텍스트 디자인 (그림자, 아웃라인, 그라데이션 효과)"""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    try:
        # 폰트 크기 조정 (더 큰 폰트)
        base_font_size = int(width * 0.08)  # 기본 크기
        
        # 한글 폰트 시도
        font_candidates = [
            "malgun.ttf",           # 윈도우 기본
            "NanumGothicBold.ttf",  # 나눔고딕 볼드
            "AppleGothic.ttf",      # 맥 기본
            "arial.ttf"             # 영문 대체
        ]
        
        font = None
        for font_name in font_candidates:
            try:
                font = ImageFont.truetype(font_name, base_font_size)
                break
            except:
                continue
        
        if not font:
            font = ImageFont.load_default()
            
    except Exception:
        font = ImageFont.load_default()
    
    # 텍스트 크기 측정
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 인물 위치에 따른 텍스트 배치
    if person_position == "center":
        # 중앙에 인물이 있으면 텍스트는 상단에
        x = (width - text_width) // 2
        y = int(height * 0.15)  # 상단 15% 지점
        
    elif person_position == "left":
        # 왼쪽에 인물이 있으면 텍스트는 오른쪽 상단에
        x = int(width * 0.55)  # 오른쪽 영역
        y = int(height * 0.2)
        
        # 텍스트가 화면을 벗어나면 조정
        if x + text_width > width * 0.95:
            x = int(width * 0.95) - text_width
            
    elif person_position == "right":
        # 오른쪽에 인물이 있으면 텍스트는 왼쪽 상단에
        x = int(width * 0.05)  # 왼쪽 영역
        y = int(height * 0.2)
        
    else:  # 기본값
        x = (width - text_width) // 2
        y = int(height * 0.15)
    
    # 🌟 고급 텍스트 효과
    # 1. 외곽선 (더 부드럽게)
    outline_width = max(4, base_font_size // 15)
    
    # 부드러운 외곽선을 위한 다중 레이어
    for layer in range(outline_width, 0, -1):
        outline_alpha = int(200 * (layer / outline_width))  # 바깥쪽일수록 투명
        outline_color = (0, 0, 0, outline_alpha)
        
        for offset_x in range(-layer, layer + 1):
            for offset_y in range(-layer, layer + 1):
                if offset_x != 0 or offset_y != 0:
                    # 투명도가 있는 검은색 외곽선
                    temp_img = Image.new('RGBA', image.size, (0, 0, 0, 0))
                    temp_draw = ImageDraw.Draw(temp_img)
                    temp_draw.text((x + offset_x, y + offset_y), text, fill=(0, 0, 0, outline_alpha), font=font)
                    image = Image.alpha_composite(image.convert('RGBA'), temp_img).convert('RGB')
    
    # 2. 메인 텍스트 (밝은 색상)
    main_color = "white"
    draw = ImageDraw.Draw(image)
    draw.text((x, y), text, fill=main_color, font=font)
    
    # 3. 하이라이트 효과
    highlight_offset = max(1, base_font_size // 30)
    highlight_color = "#FFFACD"  # 연한 노란색
    draw.text((x - highlight_offset, y - highlight_offset), text, fill=highlight_color, font=font)
    

    
    print(f"✅ 텍스트 배치 완료: {person_position} 위치 기준")
    return image
    


def enhance_background_for_composition(background_img, person_position):
    """인물 합성을 위해 배경 이미지를 향상시키는 함수"""
    try:
        from PIL import ImageEnhance, ImageFilter
        
        # 1. 전체적인 대비 향상
        enhancer = ImageEnhance.Contrast(background_img)
        background_img = enhancer.enhance(1.1)
        
        # 2. 채도 조금 증가
        enhancer = ImageEnhance.Color(background_img)
        background_img = enhancer.enhance(1.15)
        
        # 3. 인물이 들어갈 영역을 약간 어둡게 해서 인물이 돋보이도록
        overlay = Image.new('RGBA', background_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        width, height = background_img.size
        
        if person_position == "center":
            # 중앙 하단 영역을 약간 어둡게
            overlay_draw.rectangle([
                (width//4, height//2),
                (width*3//4, height)
            ], fill=(0, 0, 0, 20))
            
        elif person_position == "left":
            # 왼쪽 영역을 약간 어둡게
            overlay_draw.rectangle([
                (0, height//4),
                (width//2, height)
            ], fill=(0, 0, 0, 15))
            
        elif person_position == "right":
            # 오른쪽 영역을 약간 어둡게
            overlay_draw.rectangle([
                (width//2, height//4),
                (width, height)
            ], fill=(0, 0, 0, 15))
        
        # 오버레이 적용
        background_rgba = background_img.convert('RGBA')
        result = Image.alpha_composite(background_rgba, overlay)
        background_img = result.convert('RGB')
        
        print("✅ 배경 이미지 합성 최적화 완료")
        return background_img
        
    except Exception as e:
        print(f"배경 향상 실패: {e}")
        return background_img

# Google Cloud 설정 확인 함수
def check_google_cloud_setup():
    """Google Cloud 설정 상태 확인"""
    setup_status = {
        "project_id": bool(GOOGLE_CLOUD_PROJECT_ID),
        "credentials_file": bool(GOOGLE_APPLICATION_CREDENTIALS),
        "credentials_exists": False,
        "vertex_ai_initialized": False
    }
    
    if GOOGLE_APPLICATION_CREDENTIALS:
        setup_status["credentials_exists"] = os.path.exists(GOOGLE_APPLICATION_CREDENTIALS)
    
    try:
        aiplatform.init(project=GOOGLE_CLOUD_PROJECT_ID, location=GOOGLE_CLOUD_LOCATION)
        setup_status["vertex_ai_initialized"] = True
    except Exception:
        pass
    
    return setup_status

def combine_images(background_image, person_image, position="center", target_width=None, target_height=None):
    """배경과 인물 이미지를 합성하는 함수 (사용자 지정 사이즈로 조정)"""
    
    # 사용자가 지정한 사이즈로 배경 이미지 리사이즈
    if target_width and target_height:
        background_image = background_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # 배경 이미지 크기
    bg_width, bg_height = background_image.size
    
    # 인물 이미지 크기 조정 (위치에 따라 다르게)
    if position == "center":
        # 중앙: 좀 더 크게, 세로로 길게
        person_width = int(bg_width * 0.45)
        person_height = int(bg_height * 0.75)
        # 인물이 화면 하단에서 시작하도록
        x = (bg_width - person_width) // 2
        y = bg_height - person_height + int(bg_height * 0.1)  # 하단에서 10% 올라온 위치
        
    elif position == "left":
        # 왼쪽: 텍스트 공간을 위해 적당한 크기
        person_width = int(bg_width * 0.35)
        person_height = int(bg_height * 0.7)
        x = int(bg_width * 0.05)  # 왼쪽 가장자리에서 5% 떨어진 곳
        y = bg_height - person_height + int(bg_height * 0.05)
        
    elif position == "right":
        # 오른쪽: 텍스트 공간을 위해 적당한 크기
        person_width = int(bg_width * 0.35)
        person_height = int(bg_height * 0.7)
        x = bg_width - person_width - int(bg_width * 0.05)  # 오른쪽 가장자리에서 5% 떨어진 곳
        y = bg_height - person_height + int(bg_height * 0.05)
        
    else:  # 기본값은 center
        person_width = int(bg_width * 0.45)
        person_height = int(bg_height * 0.75)
        x = (bg_width - person_width) // 2
        y = bg_height - person_height + int(bg_height * 0.1)
    
    # 인물 이미지 리사이즈 (고품질)
    person_resized = person_image.resize((person_width, person_height), Image.Resampling.LANCZOS)
    
    # 인물 이미지에 부드러운 그림자 효과 추가
    person_with_shadow = add_soft_shadow(person_resized)
    
    # 배경에 인물 이미지 붙이기 (투명도 유지)
    background_copy = background_image.copy()
    
    # 그림자 먼저 붙이기
    shadow_offset = 8
    if x + shadow_offset < bg_width and y + shadow_offset < bg_height:
        try:
            background_copy.paste(person_with_shadow, (x + shadow_offset, y + shadow_offset), person_with_shadow)
        except:
            pass
    
    # 원본 인물 이미지 붙이기
    background_copy.paste(person_resized, (x, y), person_resized)
    
    print(f"✅ 인물 합성 완료: {position} 위치, 크기 {person_width}x{person_height}")
    return background_copy
def add_soft_shadow(image):
    """인물 이미지에 부드러운 그림자 효과 추가"""
    try:
        from PIL import ImageFilter, ImageEnhance
        
        # 그림자용 이미지 생성 (검은색)
        shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        
        # 원본 이미지의 알파 채널을 이용해 그림자 만들기
        if image.mode == 'RGBA':
            # 알파 채널 추출
            alpha = image.split()[-1]
            # 그림자 색상으로 채우기
            shadow.paste((50, 50, 50, 100), mask=alpha)  # 어두운 회색 그림자
            
            # 블러 효과 적용
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=5))
            
            return shadow
        else:
            return image
            
    except Exception as e:
        print(f"그림자 효과 추가 실패: {e}")
        return image
    

# -----------------------------
# 2. OpenAI 클라이언트 생성
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# YouTube API 라이브러리
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    print("⚠️ pip install google-api-python-client 필요")

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(title="K-POP 인기도 분석 & GPT 성공률 예측기", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class CelebrityAnalyzer:
    """간소화된 연예인 인기도 분석기"""
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
        ]
        self.session = requests.Session()
        
        # YouTube API 설정
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        self.youtube_client = None
        self.setup_youtube_api()
    
    def setup_youtube_api(self):
        """YouTube API 설정"""
        if not YOUTUBE_API_AVAILABLE:
            logger.warning("YouTube API 라이브러리 없음")
            return
        
        if not self.youtube_api_key:
            logger.warning("YOUTUBE_API_KEY 환경변수 필요")
            return
        
        try:
            self.youtube_client = build('youtube', 'v3', developerKey=self.youtube_api_key)
            logger.info("✅ YouTube API 연결 성공")
        except Exception as e:
            logger.error(f"YouTube API 설정 실패: {e}")
    
    def get_naver_popularity(self, celebrity_name: str) -> float:
        """네이버 검색 인기도 (0-100점)"""
        try:
            headers = {'User-Agent': random.choice(self.user_agents)}
            url = f"https://m.search.naver.com/search.naver?query={celebrity_name}"
            
            response = self.session.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                content = response.text
                # 간단한 인기도 측정
                content_size = len(content) / 1000
                mentions = content.lower().count(celebrity_name.lower())
                
                score = min(100, (content_size * 0.3) + (mentions * 2))
                return max(20, score)
            
        except Exception as e:
            logger.warning(f"네이버 분석 실패: {e}")
        
        return self.get_fallback_score(celebrity_name)
    
    def get_youtube_popularity(self, celebrity_name: str) -> float:
        """YouTube API를 통한 실제 인기도 분석"""
        if not self.youtube_client:
            logger.warning("YouTube API 사용 불가, 추정값 사용")
            return self.get_fallback_score(celebrity_name)
        
        try:
            # 1. 채널 검색
            channel_score = self._analyze_youtube_channels(celebrity_name)
            
            # 2. 비디오 검색  
            video_score = self._analyze_youtube_videos(celebrity_name)
            
            # 3. 최근 트렌드
            trend_score = self._analyze_recent_trends(celebrity_name)
            
            # 종합 점수 계산
            final_score = (channel_score * 0.4) + (video_score * 0.4) + (trend_score * 0.2)
            
            logger.info(f"YouTube 분석 완료: {final_score:.1f}점")
            return min(100, max(20, final_score))
            
        except Exception as e:
            logger.error(f"YouTube 분석 실패: {e}")
            return self.get_fallback_score(celebrity_name)
    
    def _analyze_youtube_channels(self, celebrity_name: str) -> float:
        """YouTube 채널 분석"""
        try:
            # 채널 검색
            search_response = self.youtube_client.search().list(
                part='snippet',
                q=f"{celebrity_name} 공식",
                type='channel',
                maxResults=5,
                regionCode='KR'
            ).execute()
            
            best_score = 30
            
            for item in search_response.get('items', []):
                channel_id = item['id']['channelId']
                
                # 채널 상세 정보
                channel_response = self.youtube_client.channels().list(
                    part='statistics',
                    id=channel_id
                ).execute()
                
                if channel_response['items']:
                    stats = channel_response['items'][0]['statistics']
                    subscriber_count = int(stats.get('subscriberCount', 0))
                    
                    # 구독자 기반 점수 계산
                    if subscriber_count > 10000000:  # 1천만 이상
                        score = 95
                    elif subscriber_count > 5000000:  # 5백만 이상
                        score = 85
                    elif subscriber_count > 1000000:  # 1백만 이상
                        score = 75
                    elif subscriber_count > 100000:  # 10만 이상
                        score = 60
                    else:
                        score = 40
                    
                    best_score = max(best_score, score)
            
            return best_score
            
        except Exception as e:
            logger.error(f"채널 분석 실패: {e}")
            return 30
    
    def _analyze_youtube_videos(self, celebrity_name: str) -> float:
        """YouTube 비디오 분석"""
        try:
            # 비디오 검색
            search_response = self.youtube_client.search().list(
                part='snippet',
                q=celebrity_name,
                type='video',
                maxResults=10,
                order='relevance',
                regionCode='KR'
            ).execute()
            
            video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            
            if not video_ids:
                return 30
            
            # 비디오 상세 정보
            videos_response = self.youtube_client.videos().list(
                part='statistics',
                id=','.join(video_ids[:10])
            ).execute()
            
            total_views = 0
            video_count = 0
            
            for video in videos_response.get('items', []):
                stats = video.get('statistics', {})
                view_count = int(stats.get('viewCount', 0))
                total_views += view_count
                video_count += 1
            
            if video_count == 0:
                return 30
            
            avg_views = total_views / video_count
            
            # 평균 조회수 기반 점수
            if avg_views > 10000000:  # 1천만뷰 이상
                return 90
            elif avg_views > 5000000:  # 5백만뷰 이상
                return 80
            elif avg_views > 1000000:  # 1백만뷰 이상
                return 70
            elif avg_views > 100000:  # 10만뷰 이상
                return 55
            else:
                return 40
                
        except Exception as e:
            logger.error(f"비디오 분석 실패: {e}")
            return 30
    
    def _analyze_recent_trends(self, celebrity_name: str) -> float:
        """최근 30일 트렌드 분석"""
        try:
            # 최근 30일 비디오 검색
            after_date = (datetime.now() - timedelta(days=30)).isoformat() + 'Z'
            
            search_response = self.youtube_client.search().list(
                part='snippet',
                q=celebrity_name,
                type='video',
                maxResults=10,
                publishedAfter=after_date,
                order='date'
            ).execute()
            
            recent_videos = len(search_response.get('items', []))
            
            # 최근 활동도 기반 점수
            if recent_videos >= 10:
                return 80
            elif recent_videos >= 5:
                return 65
            elif recent_videos >= 2:
                return 50
            else:
                return 30
                
        except Exception as e:
            logger.error(f"트렌드 분석 실패: {e}")
            return 30
    
    def get_fallback_score(self, celebrity_name: str) -> float:
        """크롤링 실패 시 추정 점수"""
        famous_celebrities = {
            "뉴진스": 90, "아이브": 88, "BTS": 95, "블랙핑크": 93,
            "르세라핌": 85, "에스파": 87, "트와이스": 82, "레드벨벳": 78,
            "스트레이키즈": 83, "세븐틴": 85, "NCT": 78, "엔하이픈": 75
        }
        
        return famous_celebrities.get(celebrity_name, 50)
    
    def analyze_celebrity(self, celebrity_name: str) -> dict:
        """연예인 종합 인기도 분석"""
        logger.info(f"🎭 {celebrity_name} 분석 시작...")
        
        # 각 플랫폼 점수
        naver_score = self.get_naver_popularity(celebrity_name)
        youtube_score = self.get_youtube_popularity(celebrity_name)
        
        # 종합 점수 (가중평균)
        overall_score = (naver_score * 0.4) + (youtube_score * 0.6)
        
        # 티어 결정
        if overall_score >= 85:
            tier = "top"
            tier_desc = "톱급 (85점 이상)"
        elif overall_score >= 70:
            tier = "major"
            tier_desc = "주요급 (70-84점)"
        elif overall_score >= 50:
            tier = "rising" 
            tier_desc = "떠오르는급 (50-69점)"
        else:
            tier = "indie"
            tier_desc = "신인급 (50점 미만)"
        
        result = {
            "name": celebrity_name,
            "naver_score": round(naver_score, 1),
            "youtube_score": round(youtube_score, 1),
            "overall_score": round(overall_score, 1),
            "tier": tier,
            "tier_description": tier_desc,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"✅ {celebrity_name} 분석 완료 - 점수: {overall_score:.1f}, 티어: {tier}")
        return result

# ==============================================
# 3. 팬덤명 검색 함수
# ==============================================

def get_fandom_name(artist_name: str) -> str:
    """아티스트명으로 팬덤명 검색"""
    
    try:
        search_prompt = f"""
        다음 아티스트의 공식 팬덤명을 알려주세요:
        아티스트명: {artist_name}
        
        팬덤명만 정확히 답해주세요. 모르면 "알 수 없음"이라고 답해주세요.
        예시:
        - BTS → 아미
        - 블랙핑크 제니 → 블링크
        - 아이유 → 유애나
        - NewJeans → 버니즈
        - 프로미스나인 -> 플로버
        왼쪽의 가수명과 화살표는 제외하고, 오른쪽의 팬덤명만 답해주세요
        답변: 
        """
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "당신은 K-POP 팬덤명에 정통한 전문가입니다. 정확한 공식 팬덤명만 제공하세요."},
                {"role": "user", "content": search_prompt}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        fandom_name = response.choices[0].message.content.strip()
        
        # 불필요한 텍스트 제거
        fandom_name = re.sub(r'^(팬덤명은\s*|답변:\s*)', '', fandom_name)
        fandom_name = re.sub(r'(입니다|이에요|예요)\.?$', '', fandom_name)
        
        return fandom_name if fandom_name and fandom_name != "알 수 없음" else "팬들"
            
    except Exception as e:
        print(f"팬덤명 검색 실패: {e}")
        return "팬들"

# ==============================================
# 4. Pydantic 모델
# ==============================================

class CampaignInput(BaseModel):
    artistName: str = Field(..., description="연예인/아이돌 이름")
    startDate: Optional[str] = Field(None, description="펀딩 시작일 (YYYY-MM-DD)")
    endDate: Optional[str] = Field(None, description="펀딩 종료일 (YYYY-MM-DD)")
    locationText: Optional[str] = Field(None, description="광고 위치")
    goal_amount_krw: Optional[int] = Field(None, description="목표금액 (원)")
    max_chars: Optional[int] = Field(default=300, description="최대 글자 수")
    brand_safety: Optional[bool] = Field(default=True, description="브랜드 세이프티 적용 여부")

    @field_validator("max_chars")
    @classmethod
    def _len_guard(cls, v):
        if v is not None and v < 80:
            raise ValueError("max_chars는 80자 이상이어야 합니다.")
        return v

class PromoOutput(BaseModel):
    text: str
    hashtags: List[str]
    text_en: str
    text_zh: str
    text_ja: str
    hashtags_en: List[str]
    hashtags_zh: List[str]
    hashtags_ja: List[str]
    used_model: str
    fandom_name: str
    token_usage: Optional[dict] = None

# ==============================================
# 5. 프롬프트 템플릿
# ==============================================

SYSTEM_KO = """당신은 팬덤 프로젝트 홍보글을 쓰는 전문 카피라이터입니다.

작성 규칙:
- 한국어로만 작성합니다.
- 간결하고 응원하는 톤을 유지하되, 과장 광고나 비속어는 금지합니다.
- 이모지는 문맥을 돋보이게 할 정도로만 사용합니다 (각 줄마다 1-2개 정도).
- 검색된 팬덤명을 반드시 사용하세요.
- 입력받은 시작날짜와 종료날짜를 정확히 "X월 X일부터 X월 X일까지" 형식으로 표시하세요.
- 광고 장소(locationText)를 "○○에서 만나요!" 형태로 포함하세요.
- 목표금액이 있으면 "목표금액 ○○원을 모읍니다" 형태로 포함하세요.
- 해시태그는 정확히 4개만 생성하세요 (아티스트명 + 광고목적 + 팬덤명 + 영어태그).

반드시 다음 형식을 따르세요:

참고 예시:
다가올 수빈의 특별한 날💙✨  
홍대입구 스크린도어 광고로 만나요!  
8월 20일부터 30일까지, MOA의 힘으로 목표금액 500만 원을 모읍니다💪  
참여해주신 분들께는 감사의 마음을 담은 특별 리워드를 드려요🎁  
#수빈생일광고 #SOOBIN_HBD #MOA와함께축하

작성 구조:
1줄: 아티스트의 특별한 날 언급 + 이모지
2줄: 광고 장소에서 만나자는 문구
3줄: 날짜 기간 + 팬덤명 + 목표금액 모금 문구
4줄: 참여 유도 문구 (리워드나 감사 표현)
5줄: 해시태그 4개
"""

TEMPLATE = """다음 정보로 팬덤 후원 홍보글을 만들어주세요.

필수 정보:
- 아티스트: {artist_name}
- 팬덤명: {fandom_name} (반드시 사용)
- 펀딩 기간: {start_date}부터 {end_date}까지 (정확한 날짜 형식으로)
- 광고 장소: {locationText} (여기서 만나요 형태로)
- 목표금액: {goal_amount}원

작성 요구사항:
1) 최대 {max_chars}자 내에서 작성
2) 첫 줄: 아티스트의 특별한 날을 축하하는 문구 + 적절한 이모지
3) 둘째 줄: "{locationText}에서 만나요!" 형태로 장소 표시
4) 셋째 줄: "{start_date}부터 {end_date}까지, {fandom_name}의 힘으로 목표금액 {goal_amount}원을 모읍니다" 형태
5) 넷째 줄: 참여 유도 문구 (감사 표현이나 리워드 언급)
6) 다섯째 줄: 해시태그 정확히 4개 (#아티스트명+행사 #영어태그 #팬덤명+키워드 #추가키워드)

출력 형식:
---
[홍보글]
(1줄: 특별한 날 축하 문구)
(2줄: 장소 만나요 문구)  
(3줄: 기간 + 팬덤명 + 목표금액 문구)
(4줄: 참여 유도 문구)

[해시태그]
#태그1 #태그2 #태그3 #태그4
---
"""

def build_user_prompt(artist_name: str, **kwargs) -> tuple[str, str]:
    """프롬프트 생성 및 팬덤명 검색"""
    
    # 팬덤명 자동 검색
    fandom_name = get_fandom_name(artist_name)
    
    # 날짜 포맷팅 (YYYY-MM-DD -> M월 D일)
    def format_date(date_str):
        if not date_str or date_str == "없음":
            return "미정"
        try:
            from datetime import datetime
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return f"{date_obj.month}월 {date_obj.day}일"
        except:
            return date_str
    
    # 금액 포맷팅
    def format_amount(amount):
        if not amount:
            return "미정"
        if amount >= 10000:
            return f"{amount // 10000}만"
        else:
            return str(amount)
    
    start_date = format_date(kwargs.get("startDate"))
    end_date = format_date(kwargs.get("endDate"))
    goal_amount = format_amount(kwargs.get("goal_amount_krw"))
    locationText = kwargs.get("locationText") or "광고 장소"
    
    prompt = TEMPLATE.format(
        artist_name=artist_name,
        fandom_name=fandom_name,
        start_date=start_date,
        end_date=end_date,
        locationText=locationText,
        goal_amount=goal_amount,
        max_chars=kwargs.get("max_chars", 300)
    )
    
    return prompt, fandom_name

# ==============================================
# 6. 다국어 번역 함수
# ==============================================

def translate_promo_text(korean_text: str, korean_hashtags: List[str], artist_name: str, fandom_name: str) -> Dict[str, Any]:
    """한국어 홍보글을 영어, 중국어, 일본어로 번역"""
    
    translation_prompt = f"""
다음 K-POP 팬덤 홍보글을 영어, 중국어(간체), 일본어로 번역해주세요.
각 언어별로 자연스럽고 현지 팬덤 문화에 맞게 번역하세요.

원본 한국어 홍보글:
{korean_text}

원본 해시태그: {' '.join(korean_hashtags)}

번역시 주의사항:
- 아티스트명: {artist_name}
- 팬덤명: {fandom_name}
- 각 언어권 팬덤 문화에 맞는 톤앤매너 사용
- 이모지는 원본과 유사하게 유지
- 해시태그는 각 언어에 맞게 적절히 번역/현지화

출력 형식:
---
[ENGLISH]
(영어 번역본)

[ENGLISH_HASHTAGS]
#tag1 #tag2 #tag3 #tag4

[CHINESE]
(중국어 번역본)

[CHINESE_HASHTAGS]
#标签1 #标签2 #标签3 #标签4

[JAPANESE]
(일본어 번역본)

[JAPANESE_HASHTAGS]
#タグ1 #タグ2 #タグ3 #タグ4
---
"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "당신은 K-POP 팬덤 문화에 정통한 다국어 번역 전문가입니다. 각 언어권의 팬덤 문화와 표현 방식에 맞게 자연스럽게 번역하세요."},
                {"role": "user", "content": translation_prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        
        # 번역 결과 파싱
        translations = {
            "text_en": "",
            "text_zh": "",
            "text_ja": "",
            "hashtags_en": [],
            "hashtags_zh": [],
            "hashtags_ja": []
        }
        
        # 영어 추출
        if "[ENGLISH]" in content and "[ENGLISH_HASHTAGS]" in content:
            translations["text_en"] = content.split("[ENGLISH]")[1].split("[ENGLISH_HASHTAGS]")[0].strip()
            en_tags = content.split("[ENGLISH_HASHTAGS]")[1].split("[CHINESE]")[0].strip()
            translations["hashtags_en"] = [tag for tag in en_tags.split() if tag.startswith("#")]
        
        # 중국어 추출
        if "[CHINESE]" in content and "[CHINESE_HASHTAGS]" in content:
            translations["text_zh"] = content.split("[CHINESE]")[1].split("[CHINESE_HASHTAGS]")[0].strip()
            zh_tags = content.split("[CHINESE_HASHTAGS]")[1].split("[JAPANESE]")[0].strip()
            translations["hashtags_zh"] = [tag for tag in zh_tags.split() if tag.startswith("#")]
        
        # 일본어 추출
        if "[JAPANESE]" in content and "[JAPANESE_HASHTAGS]" in content:
            translations["text_ja"] = content.split("[JAPANESE]")[1].split("[JAPANESE_HASHTAGS]")[0].strip()
            ja_tags = content.split("[JAPANESE_HASHTAGS]")[1].strip()
            translations["hashtags_ja"] = [tag for tag in ja_tags.split() if tag.startswith("#")]
        
        return translations
        
    except Exception as e:
        print(f"번역 실패: {e}")
        return {
            "text_en": "Translation unavailable",
            "text_zh": "翻译不可用",
            "text_ja": "翻訳利用不可",
            "hashtags_en": ["#translation_error"],
            "hashtags_zh": ["#翻译错误"],
            "hashtags_ja": ["#翻訳エラー"]
        }

# ==============================================
# 7. 응답 파서
# ==============================================

def parse_response_text(text: str) -> Dict[str, Any]:
    body = ""
    tags: List[str] = []
    if "[홍보글]" in text and "[해시태그]" in text:
        body = text.split("[홍보글]")[-1].split("[해시태그]")[0].strip()
        tag_part = text.split("[해시태그]")[-1].strip()
        tags = [w for w in tag_part.replace("\n", " ").split(" ") if w.startswith("#")]
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        hash_line = next((ln for ln in reversed(lines) if ln.startswith("#")), "")
        if hash_line:
            body_lines = lines[: lines.index(hash_line)]
            body = "\n".join(body_lines)
            tags = [w for w in hash_line.split(" ") if w.startswith("#")]
        else:
            body = text.strip()

    tags = list(dict.fromkeys(tags))[:4]  # 정확히 4개로 제한
    return {"text": body, "hashtags": tags}


class GPTPredictor:
    """GPT를 활용한 성공률 예측기"""
    
    def __init__(self):
        self.setup_openai()
    
    def setup_openai(self):
        """OpenAI 설정"""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key
            self.openai_available = True
            logger.info("✅ OpenAI 연결 성공")
        else:
            self.openai_available = False
            logger.warning("⚠️ OPENAI_API_KEY 설정 필요")
    
    def predict_success_rate(self, celebrity_data: dict, campaign_info: dict) -> dict:
        """GPT를 활용한 성공률 예측"""
        if not self.openai_available:
            return {
                "success": False,
                "message": "OpenAI API 키가 필요합니다"
            }
        
        try:
            # GPT 프롬프트 생성
            prompt = f"""
당신은 K-POP 마케팅 전문가입니다. 지하철 광고 캠페인 성공률을 예측해주세요.

연예인 정보:
- 이름: {celebrity_data['name']}
- 종합 인기도: {celebrity_data['overall_score']}/100점
- 티어: {celebrity_data['tier_description']}
- 네이버 점수: {celebrity_data['naver_score']}/100점
- 유튜브 점수: {celebrity_data['youtube_score']}/100점

캠페인 정보:
- 목표 금액: {campaign_info['target_amount']:,}원
- 모금 기간: {campaign_info['duration_days']}일
- 광고 위치: {campaign_info['location']}
- 광고 목적: {campaign_info['purpose']}

위 정보를 바탕으로 다음을 JSON 형태로 예측해주세요:
{{
    "success_rate": 예측_성공률_숫자만(0-100),
    "confidence": "높음/보통/낮음",
    "analysis": "예측 근거 설명 (200자 이내)",
    "recommendations": ["추천사항1", "추천사항2", "추천사항3"],
    "risk_factors": ["리스크1", "리스크2"]
}}
"""

            # GPT 호출
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 K-POP 마케팅 전문가입니다. JSON 형태로만 답변하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            # 응답 파싱
            response_text = response.choices[0].message.content
            gpt_result = json.loads(response_text)
            
            return {
                "success": True,
                "prediction": gpt_result,
                "model_used": "gpt-4o-mini"
            }
            
        except Exception as e:
            logger.error(f"GPT 예측 실패: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# 전역 인스턴스
analyzer = CelebrityAnalyzer()
predictor = GPTPredictor()


# 🚀 메인 API: 광고 이미지 생성
@app.post("/generate-ad-image")
async def generate_ad_image(
    celebrity_image: UploadFile = File(...),  # 연예인 사진 파일
    background_color: str = Form(...),        # 배경색 선택
    mood: str = Form(...),                    # 분위기 선택
    effects: str = Form(...),                 # 효과 선택
    width: int = Form(...),                   # 🆕 사용자가 직접 입력한 가로 크기
    height: int = Form(...),                  # 🆕 사용자가 직접 입력한 세로 크기
    custom_text: str = Form(...),             # 사용자 입력 문구
    position: str = Form("center")            # 인물 위치 (center/left/right)
):
    """
    광고 이미지를 생성하는 메인 API
    사용자가 입력한 크기로 맞춤형 광고 생성
    """
    try:
        # 입력 검증
        if width <= 0 or height <= 0:
            return {"success": False, "message": "가로, 세로 크기는 0보다 커야 합니다"}
        
        if width > 3000 or height > 3000:
            return {"success": False, "message": "최대 크기는 3000px입니다"}
        
        # 위치 검증
        valid_positions = ["center", "left", "right"]
        if position not in valid_positions:
            position = "center"
        
        
        # 1단계: 업로드된 연예인 사진에서 배경 제거
        print("1단계: 배경 제거 중...")
        person_no_bg = remove_background_from_image(celebrity_image.file)
        
        # 2단계: AI 프롬프트 생성 (사용자 지정 크기 고려)
        print("2단계: AI 프롬프트 생성 중...")
        prompt = generate_prompt(background_color, mood, effects, width, height)
        
        print(f"🎨 프롬프트 생성 완료!")
        
        # 3단계: AI로 배경 이미지 생성
        print("3단계: AI 배경 생성 중...")
        background_img = await generate_background_with_ai(prompt)
        
        # 4단계: 배경과 인물 이미지 합성 (사용자 지정 사이즈로 조정)
        print("4단계: 이미지 합성 중...")
        final_image = combine_images(background_img, person_no_bg, position, width, height)
        
        # 5단계: 텍스트 추가
        print("5단계: 텍스트 추가 중...")
        final_image = add_text_to_image(final_image, custom_text, "top")
        
        # 6단계: 결과 이미지 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{width}x{height}_{timestamp}.png"
        filepath = f"generated_ads/{filename}"
        final_image.save(filepath)
        
        print(f"✅ 생성 완료: {filename}")
        
        # 7단계: 결과 반환
        return {
            "success": True,
            "message": f"생성 완료!",
            "dimensions": {"width": width, "height": height},
            "person_position": position,
            "filename": filename,
            "download_url": f"/download/{filename}",
            "composition_info": {
                "background_enhanced": True,
                "shadow_applied": True,
                "text_position_optimized": True,
                "flexible_color_processing": True
            }
        }
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return {
            "success": False,
            "message": f"이미지 생성 실패: {str(e)}"
        }

# 생성된 이미지 다운로드용 API
@app.get("/download/{filename}")
async def download_image(filename: str):
    """생성된 광고 이미지를 다운로드할 수 있게 해주는 API"""
    filepath = f"generated_ads/{filename}"
    if os.path.exists(filepath):
        return FileResponse(filepath)
    else:
        return {"error": "파일을 찾을 수 없습니다"}
    
# API 엔드포인트들
@app.get("/")
async def root():
    return {
        "service": "K-POP 인기도 분석 & GPT 성공률 예측기",
        "version": "1.0",
        "main_endpoint": "POST /predict - 연예인 분석 + 캠페인 성공률 예측"
    }

@app.post("/predict")
async def predict_campaign(
    celebrity_name: str = Form(...),
    target_amount: int = Form(...),
    duration_days: int = Form(...),
    location: str = Form(...),
    purpose: str = Form(...)
):
    """연예인 분석 + 캠페인 성공률 예측 (원스톱)"""
    try:
        # 1. 연예인 분석
        celebrity_data = analyzer.analyze_celebrity(celebrity_name)
        
        # 2. 캠페인 정보
        campaign_info = {
            "target_amount": target_amount,
            "duration_days": duration_days,
            "location": location,
            "purpose": purpose
        }
        
        # 3. GPT 예측
        prediction_result = predictor.predict_success_rate(celebrity_data, campaign_info)
        
        # 4. 통합 결과
        if prediction_result["success"]:
            prediction_data = prediction_result["prediction"]
            
            return {
                "success": True,
                "celebrity_name": celebrity_name,
                "celebrity_analysis": {
                    "overall_score": celebrity_data["overall_score"],
                    "tier": celebrity_data["tier_description"],
                    "naver_score": celebrity_data["naver_score"],
                    "youtube_score": celebrity_data["youtube_score"]
                },
                "campaign_info": campaign_info,
                "prediction": {
                    "success_rate": f"{prediction_data['success_rate']}%",
                    "confidence": prediction_data["confidence"],
                    "analysis": prediction_data["analysis"],
                    "recommendations": prediction_data["recommendations"],
                    "risk_factors": prediction_data.get("risk_factors", [])
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            return prediction_result
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/v1/promo:generate", response_model=PromoOutput)
def generate_promo(payload: CampaignInput):
    # 프롬프트 생성 및 팬덤명 검색
    payload_dict = payload.model_dump()
    user_prompt, fandom_name = build_user_prompt(
        payload_dict["artistName"], 
        **{k: v for k, v in payload_dict.items() if k != 'artistName'}
    )

    try:
        # 한국어 홍보글 생성
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_KO},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=600,
            temperature=0.7,
        )
        
        # 한국어 홍보글 파싱
        text = resp.choices[0].message.content
        parsed = parse_response_text(text)

        # 다국어 번역 수행
        translations = translate_promo_text(
            parsed["text"], 
            parsed["hashtags"], 
            payload_dict["artistName"], 
            fandom_name
        )

        # 브랜드 안전성 검사
        if payload.brand_safety:
            banned = ["협박", "비방", "증오", "차별"]
            if any(b in parsed["text"] for b in banned):
                raise HTTPException(status_code=400, detail="부적절한 표현이 감지되었습니다.")

        # 글자 수 제한
        if payload.max_chars and len(parsed["text"]) > payload.max_chars:
            parsed["text"] = parsed["text"][: payload.max_chars].rstrip() + "…"

        # 토큰 사용량 정보 추출
        token_usage = None
        if hasattr(resp, 'usage') and resp.usage:
            try:
                token_usage = {
                    "prompt_tokens": getattr(resp.usage, 'prompt_tokens', None),
                    "completion_tokens": getattr(resp.usage, 'completion_tokens', None),
                    "total_tokens": getattr(resp.usage, 'total_tokens', None),
                }
            except:
                token_usage = {"error": "토큰 정보 추출 실패"}

        return PromoOutput(
            text=parsed["text"],
            hashtags=parsed["hashtags"],
            text_en=translations["text_en"],
            text_zh=translations["text_zh"],
            text_ja=translations["text_ja"],
            hashtags_en=translations["hashtags_en"],
            hashtags_zh=translations["hashtags_zh"],
            hashtags_ja=translations["hashtags_ja"],
            used_model=MODEL,
            fandom_name=fandom_name,
            token_usage=token_usage,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"생성 실패: {e}")


@app.get("/status")
async def get_status():
    """시스템 상태 확인"""
    return {
        "youtube_api": "연결됨" if analyzer.youtube_client else "설정 필요",
        "openai_api": "연결됨" if predictor.openai_available else "설정 필요",
        "required_env_vars": {
            "YOUTUBE_API_KEY": "YouTube Data API v3 키",
            "OPENAI_API_KEY": "OpenAI API 키"
        }
    }

