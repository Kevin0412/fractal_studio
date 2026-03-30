<template>
  <section class="view-card">
    <h3>{{ t('formulas') }}</h3>
    <ul>
      <li v-for="item in formulas" :key="item.name">
        <strong>{{ item.name }}:</strong> {{ item.formula }}
      </li>
    </ul>
  </section>
</template>

<script setup lang="ts">
import { t } from '../i18n'

const formulas = [
  { name: 'Classic Mandelbrot', formula: 'z_{n+1}=z_n^2+c' },
  { name: 'Conjugate Mandelbrot', formula: 'z_{n+1}=conj(z_n^2)+c' },
  { name: 'Burning Ship', formula: 'z_{n+1}=(|Re(z_n)|+i|Im(z_n)|)^2+c' },
  { name: 'Duck Fold', formula: 'z_{n+1}=(Re(z_n)+i|Im(z_n)|)^2+c' },
  { name: 'Bell Mirror', formula: 'z_{n+1}=(|Re(z_n)|-iIm(z_n))^2+c' },
  { name: 'Fish Tail', formula: 'z_{n+1}=|Re(z_n^2)|+iIm(z_n^2)+c' },
  { name: 'Vase Mirror', formula: 'z_{n+1}=|Re(z_n^2)|-iIm(z_n^2)+c' },
  { name: 'Bird Wing', formula: 'z_{n+1}=|Re(z_n^2)|+i|Im(z_n^2)|+c' },
  { name: 'Mask Fold', formula: 'z_{n+1}=|Re((Re(z_n)+i|Im(z_n)|)^2)|+iIm((Re(z_n)+i|Im(z_n)|)^2)+c' },
  { name: 'Ship Fold', formula: 'z_{n+1}=|Re((|Re(z_n)|+iIm(z_n))^2)|-iIm((|Re(z_n)|+iIm(z_n))^2)+c' },
]
</script>
