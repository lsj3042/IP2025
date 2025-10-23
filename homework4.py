import numpy as np
import cv2

def run_main():
    # 📌 이미지 파일 불러오기
    frame = cv2.imread('sIMG_8253.JPG')

    if frame is None:
        print("이미지를 불러올 수 없습니다. 파일 경로를 확인하세요.")
        return

    # ROI 설정 (필요시 크기 조정)
    roi = frame[0:700, 0:700]

    # 그레이스케일 변환
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 🔧 블러 적용 (디테일 보존)
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 🔧 적응형 이진화 (파라미터 조정)
    thresh = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        17, 2
    )

    # 🔧 모폴로지 닫기 연산 (윤곽선 붙이기)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

    # 윤곽선 찾기
    cont_img = closing.copy()
    contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 🔍 디버깅용: 윤곽선 그리기 (선택)
    # cv2.drawContours(roi, contours, -1, (255, 0, 0), 1)

    # 윤곽선 필터링 및 타원 그리기
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 🔧 면적 범위 확장
        if area < 500 or area > 15000:
            continue

        if len(cnt) < 5:
            continue

        # 타원 fitting 및 그리기
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(roi, ellipse, (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow("Morphological Closing", closing)
    cv2.imshow("Adaptive Thresholding", thresh)
    cv2.imshow("Contours", roi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_main()
