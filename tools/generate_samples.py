from pathlib import Path
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import qrcode

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data' / 'samples'
OUT.mkdir(parents=True, exist_ok=True)
W,H = 900,1200
QR_PAYLOAD = 'https://example.org/banana-hacks/registration?event=PP-DEMO'
QR_BAD = 'https://example.org/unexpected?source=changed-demo'
FONT_CANDIDATES = [
    Path('/usr/share/fonts/truetype/dejavu'),
    Path('/usr/share/fonts/truetype/liberation2'),
    Path('C:/Windows/Fonts'),
    Path('/System/Library/Fonts'),
]
FONT_ALIASES = {
    'DejaVuSans.ttf': ['DejaVuSans.ttf', 'arial.ttf', 'LiberationSans-Regular.ttf', 'SegoeUI.ttf'],
    'DejaVuSans-Bold.ttf': ['DejaVuSans-Bold.ttf', 'arialbd.ttf', 'LiberationSans-Bold.ttf', 'SegoeUIb.ttf']
}

def font(name, size):
    candidates = FONT_ALIASES.get(name, [name])
    for folder in FONT_CANDIDATES:
        for font_name in candidates:
            candidate = folder / font_name
            if candidate.exists():
                try:
                    return ImageFont.truetype(str(candidate), size)
                except Exception:
                    pass
    # Fallback keeps the fixture generator functional on minimal systems with requested size.
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()

REGULAR='DejaVuSans.ttf'; BOLD='DejaVuSans-Bold.ttf'

def make_qr(payload):
    q=qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=4)
    q.add_data(payload); q.make(fit=True)
    im=q.make_image(fill_color='black', back_color='white').convert('RGB')
    return im.resize((190,190), Image.Resampling.NEAREST)

def make_poster(date='10 OCTOBER 2026', payload=QR_PAYLOAD, title='BANANA HACKS', subtitle='BUILD SOMETHING THAT MATTERS'):
    im=Image.new('RGB',(W,H),(248,250,253)); d=ImageDraw.Draw(im)
    # Strong, repeatable visual landmarks for registration.
    d.rectangle((0,0,W,24), fill=(31,57,101))
    d.rectangle((42,52,858,1148), outline=(31,57,101), width=5)
    d.rectangle((55,65,845,270), fill=(23,38,66))
    d.ellipse((700,75,820,195), fill=(65,125,225))
    d.ellipse((745,112,790,157), fill=(248,188,73))
    d.text((90,91),'POSTERPROOF DEMO',font=font(BOLD,22),fill=(164,194,255))
    d.text((88,134),title,font=font(BOLD,58),fill=(255,255,255),stroke_width=0)
    d.text((91,213),subtitle,font=font(REGULAR,19),fill=(225,233,247))
    # Geometric marks, dividers, and small labels provide useful feature texture.
    d.polygon([(63,296),(837,296),(816,319),(84,319)], fill=(68,119,213))
    d.text((120,350),'CAMPUS COMMUNITY EVENT',font=font(BOLD,19),fill=(63,81,105))
    d.text((120,383),'Meet builders, share ideas, create prototypes.',font=font(REGULAR,22),fill=(63,81,105))
    # Monitored event date field. Coordinates: x=120, y=425, w=650, h=135.
    d.rounded_rectangle((120,425,770,560),radius=12,fill=(231,239,253),outline=(67,112,196),width=3)
    d.text((145,440),'EVENT DATE',font=font(BOLD,17),fill=(45,78,139))
    d.text((145,478),date,font=font(BOLD,35),fill=(22,36,62))
    d.text((120,600),'WHAT TO EXPECT',font=font(BOLD,20),fill=(31,57,101))
    # Three compact cards with distinct shapes/icons.
    cards=[(120,642,320,755,'01','BUILD'),(350,642,550,755,'02','LEARN'),(580,642,780,755,'03','SHARE')]
    for x1,y1,x2,y2,n,label in cards:
        d.rounded_rectangle((x1,y1,x2,y2),radius=11,fill=(255,255,255),outline=(188,202,222),width=2)
        d.ellipse((x1+16,y1+17,x1+50,y1+51),fill=(65,125,225))
        d.text((x1+25,y1+22),n,font=font(BOLD,12),fill='white')
        d.text((x1+61,y1+22),label,font=font(BOLD,19),fill=(31,57,101))
        d.line((x1+18,y1+69,x2-18,y1+69),fill=(221,228,239),width=2)
        d.line((x1+18,y1+83,x2-55,y1+83),fill=(221,228,239),width=2)
    d.text((120,785),'SCAN TO VIEW THE SAMPLE REGISTRATION',font=font(BOLD,15),fill=(63,81,105))
    im.paste(make_qr(payload),(650,820))
    d.text((120,850),'DEMO QR · EXAMPLE.ORG',font=font(BOLD,16),fill=(31,57,101))
    d.text((120,883),'Synthetic payload for prototype testing only.',font=font(REGULAR,15),fill=(92,108,130))
    # Footer lines and ornamental squares.
    d.line((120,1002,780,1002),fill=(196,208,226),width=2)
    d.text((120,1028),'REFERENCE VERSION: PP-DEMO-001',font=font(BOLD,16),fill=(44,62,88))
    d.text((120,1062),'Illustrative artwork. Not an official event poster.',font=font(REGULAR,15),fill=(92,108,130))
    for i in range(8):
        d.rectangle((120+i*24,1100,132+i*24,1112),fill=(65,125,225) if i%2==0 else (248,188,73))
    return im

def perspective_photo(poster, seed=11):
    rng=np.random.default_rng(seed)
    arr=np.zeros((1350,1100,3),dtype=np.uint8)
    # neutral tabletop/background with subtle texture
    arr[:]=np.array([224,226,229],dtype=np.uint8)
    noise=rng.normal(0,2.0,arr.shape[:2]).astype(np.int16)
    for c in range(3): arr[:,:,c]=np.clip(arr[:,:,c].astype(np.int16)+noise,0,255).astype(np.uint8)
    # Fine background cues outside the sheet (not part of poster).
    bg=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)); bd=ImageDraw.Draw(bg)
    bd.line((25,190,95,230),fill=(199,203,207),width=3)
    bd.ellipse((980,105,1035,160),outline=(188,193,199),width=3)
    arr=cv2.cvtColor(np.array(bg),cv2.COLOR_RGB2BGR)
    src=np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    # Distinct quadrilateral for each generated photo.
    jitter=rng.integers(-18,19,(4,2)).astype(np.float32)
    dst=np.float32([[118,100],[965,74],[1005,1260],[85,1214]])+jitter
    M=cv2.getPerspectiveTransform(src,dst)
    warped=cv2.warpPerspective(cv2.cvtColor(np.array(poster),cv2.COLOR_RGB2BGR),M,(1100,1350))
    mask=cv2.warpPerspective(np.full((H,W),255,np.uint8),M,(1100,1350))
    arr[mask>0]=warped[mask>0]
    # Slight sensor-like noise and gentle blur to simulate a phone capture.
    grain=rng.normal(0,1.2,arr.shape).astype(np.int16)
    arr=np.clip(arr.astype(np.int16)+grain,0,255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB))

make_poster().save(OUT/'approved_master.png')
perspective_photo(make_poster(), seed=11).save(OUT/'poster_unchanged_photo.png')
perspective_photo(make_poster(date='17 DECEMBER 2027'), seed=17).save(OUT/'poster_changed_date_photo.png')
perspective_photo(make_poster(payload=QR_BAD), seed=23).save(OUT/'poster_changed_qr_photo.png')
# Unrelated sample uses dissimilar layout and texture to exercise conservative refusal.
other=Image.new('RGB',(1100,1350),(224,235,224)); od=ImageDraw.Draw(other)
for i in range(30):
    x=(i*83)%1100; y=(i*137)%1350
    od.ellipse((x,y,x+70,y+70),fill=((60+i*3)%200+30,(90+i*7)%160+50,(80+i*5)%160+50))
for i,txt in enumerate(['COMMUNITY GARDEN','WEEKEND MARKET','LOCAL FOOD','MUSIC & ART','VISIT US']):
    od.text((60,180+i*175),txt,font=font(BOLD,45),fill=(25,55,60))
    od.line((60,245+i*175,1000,245+i*175),fill=(45,115,80),width=5)
other.save(OUT/'unrelated_poor_input.png')
print('Generated samples in',OUT)
for p in sorted(OUT.iterdir()): print(p.name, p.stat().st_size)
