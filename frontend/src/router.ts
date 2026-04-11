import { createRouter, createWebHistory } from 'vue-router'

import MapView    from './views/MapView.vue'
import PointsView from './views/PointsView.vue'
import ThreeDView from './views/ThreeDView.vue'
import RunsView   from './views/RunsView.vue'
import SystemView from './views/SystemView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/',       name: 'map',    component: MapView },
    { path: '/points', name: 'points', component: PointsView },
    { path: '/3d',     name: '3d',     component: ThreeDView },
    { path: '/runs',   name: 'runs',   component: RunsView },
    { path: '/system', name: 'system', component: SystemView },
  ],
})

export default router
