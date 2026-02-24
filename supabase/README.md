# DFA-Auto-Flow Supabase Samples

Samples 데이터는 Life Matrix Flow / ML Auto Flow와 **동일한 Supabase 프로젝트**의 `autoflow_samples` 스키마를 사용합니다.

- **대분류(app_section)**: `DFA`
- 스키마가 이미 적용되어 있다면 추가 마이그레이션 없이 시드만 실행하면 됩니다.

## 시드 실행

`.env`에 `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`(또는 `NEXT_PUBLIC_*`) 설정 후:

```bash
node scripts/seed-dfa-samples-to-supabase.mjs
```

또는 (스크립트 등록 시): `pnpm run seed:samples`
