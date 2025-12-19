// 브라우저 콘솔에서 실행할 스크립트
// 현재 모델을 샘플 형식으로 가져오기

// App.tsx의 React 컴포넌트에서 modules와 connections를 가져오는 방법
// 이 스크립트는 브라우저 콘솔에서 실행해야 합니다.

// 방법 1: React DevTools를 통해 접근 (개발 모드에서만 작동)
// 방법 2: localStorage에서 가져오기 (이미 저장된 경우)
// 방법 3: window 객체에 모델을 노출시키는 코드 추가

// 임시로 window 객체에 모델을 노출시키는 코드를 App.tsx에 추가해야 합니다.
// 또는 브라우저에서 "현재 모델 저장"을 클릭한 후 localStorage에서 가져올 수 있습니다.

// localStorage에서 모델 가져오기
const myWorkModels = JSON.parse(localStorage.getItem('myWorkModels') || '[]');
const initialModel = JSON.parse(localStorage.getItem('initialModel') || 'null');

console.log('My Work Models:', myWorkModels);
console.log('Initial Model:', initialModel);

// 가장 최근에 저장된 모델 사용
const latestModel = myWorkModels.length > 0 ? myWorkModels[myWorkModels.length - 1] : initialModel;

if (latestModel) {
  const sampleFormat = {
    name: latestModel.name || "XoL Experience",
    modules: latestModel.modules,
    connections: latestModel.connections
  };
  
  console.log('Sample Format:');
  console.log(JSON.stringify(sampleFormat, null, 2));
  
  // 클립보드에 복사
  navigator.clipboard.writeText(JSON.stringify(sampleFormat, null, 2)).then(() => {
    console.log('✓ 샘플 형식이 클립보드에 복사되었습니다!');
  }).catch(err => {
    console.error('클립보드 복사 실패:', err);
    console.log('위의 JSON을 수동으로 복사하세요.');
  });
} else {
  console.log('저장된 모델이 없습니다. 먼저 "현재 모델 저장"을 클릭하세요.');
}

