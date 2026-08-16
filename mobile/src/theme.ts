/** Shared design tokens. Small on purpose -- no UI library. */

export const colors = {
  background: '#F7F7F5',
  surface: '#FFFFFF',
  border: '#E3E3DF',
  text: '#1A1A18',
  textMuted: '#6B6B66',
  primary: '#1F5C3D',
  primaryText: '#FFFFFF',

  // One colour per item status, used consistently for pills and card accents
  // so auto / review / unmatched are distinguishable at a glance.
  auto: '#1F5C3D',
  autoSoft: '#E6F1EA',
  review: '#8A5A00',
  reviewSoft: '#FBF0DC',
  unmatched: '#6B6B66',
  unmatchedSoft: '#ECECE8',

  danger: '#A3271F',
  dangerSoft: '#FBE9E7',
};

export const spacing = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32 };

export const radius = { sm: 6, md: 10, lg: 14 };

export const statusPalette = {
  auto: { fg: colors.auto, bg: colors.autoSoft, label: 'High confidence' },
  review: { fg: colors.review, bg: colors.reviewSoft, label: 'Needs review' },
  unmatched: { fg: colors.unmatched, bg: colors.unmatchedSoft, label: 'No match' },
} as const;
