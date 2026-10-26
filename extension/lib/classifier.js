// Нормировка признаков и логистическая регрессия: P(ИИ) = σ(w·z + b), формула (2.5) диплома.
export const FEATURE_LABELS = {
  mean_sent_len: 'средняя длина предложения',
  cv_sent_len: 'разброс длины предложений',
  mean_word_len: 'средняя длина слова',
  ttr: 'разнообразие словаря (TTR)',
  punct_ratio: 'доля знаков препинания',
  func_ratio: 'доля служебных слов',
  rep_bigram_ratio: 'повторы пар слов',
  rep_trigram_ratio: 'повторы троек слов',
  comma_per_sent: 'запятых на предложение',
  mean_lp: 'предсказуемость слов для модели',
  std_lp: 'разброс предсказуемости',
  mean_entropy: 'неопределённость модели',
  mean_log_rank: 'средний ранг слов в прогнозе модели',
  rank_1: 'доля слов — первых в прогнозе',
  rank_2_10: 'доля слов из топ-10 прогноза',
  rank_11_100: 'доля слов из топ-100 прогноза',
  rank_101_1000: 'доля слов из топ-1000 прогноза',
  rank_gt_1000: 'доля редких для модели слов',
  curvature: 'кривизна вероятности (как в Fast-DetectGPT)',
};

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export function classify(features, model) {
  let z = model.intercept;
  const contributions = [];
  model.features.forEach((name, i) => {
    const x = (features[name] - model.mean[i]) / (model.scale[i] || 1);
    const c = model.coef[i] * x;
    z += c;
    contributions.push({ name, label: FEATURE_LABELS[name] || name, contribution: c });
  });
  contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return { p: sigmoid(z), top: contributions.slice(0, 3) };
}
