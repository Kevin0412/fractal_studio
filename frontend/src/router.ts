import { createRouter, createWebHistory } from 'vue-router'

import DashboardView from './views/DashboardView.vue'
import AtlasView from './views/AtlasView.vue'
import SpecialPointsView from './views/SpecialPointsView.vue'
import TransitionConversionView from './views/TransitionConversionView.vue'
import HiddenStructureView from './views/HiddenStructureView.vue'
import ThreeDFamilyView from './views/ThreeDFamilyView.vue'
import StlGalleryView from './views/StlGalleryView.vue'
import ConclusionsView from './views/ConclusionsView.vue'
import ExplorerMapView from './views/ExplorerMapView.vue'
import ArtifactsView from './views/ArtifactsView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'dashboard', component: DashboardView },
    { path: '/atlas', name: 'atlas', component: AtlasView },
    { path: '/special-points', name: 'special-points', component: SpecialPointsView },
    { path: '/transition-conversion', name: 'transition-conversion', component: TransitionConversionView },
    { path: '/hidden-structure', name: 'hidden-structure', component: HiddenStructureView },
    { path: '/hidden-structure-families', name: 'hidden-structure-families', component: ThreeDFamilyView },
    { path: '/stl-gallery', name: 'stl-gallery', component: StlGalleryView },
    { path: '/explorer-map', name: 'explorer-map', component: ExplorerMapView },
    { path: '/artifacts', name: 'artifacts', component: ArtifactsView },
    { path: '/conclusions', name: 'conclusions', component: ConclusionsView },
  ],
})

export default router
