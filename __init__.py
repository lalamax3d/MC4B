# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name" : "MC4B",
    "author" : "lalamax3d",
    "description" : "Motion Capture For Blender - Just for fun",
    "blender" : (2, 80, 0),
    "version" : (0, 0, 3),
    "location" : "View3D",
    "warning" : "",
    "category" : "Animation"
}

import bpy
from bpy import context as context

from . OpenCVAnimOperator import state
from . OpenCVAnimOperator import OpenCVAnimOperator, MC4BPropGrp

# SPECIAL LINE
bpy.types.Scene.ff_MC4B_prop_grp = bpy.props.PointerProperty(type=MC4BPropGrp)




# MAIN PANEL CONTROL
class MC4B_PT_Panel(bpy.types.Panel):
    bl_idname = "MC4B_PT_Panel"
    bl_label = "Mocap For Fun"
    bl_category = "FF_Tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    def draw(self,context):
        #active_obj = context.active_object
        layout = self.layout
        s = state()
        col0 = layout.column(align = True)
        col0.label(text='RigSetting')
        row = col0.row(align = True)
        row.prop(s, 'Src_Rig', text='Rig Src', icon='ARMATURE_DATA')

        col = layout.column(align=1)
        row = col.row(align = True)
        row.operator("wm.opencv_operator", text="Start Camera")







classes = (
        OpenCVAnimOperator,
        MC4B_PT_Panel)


register,unregister = bpy.utils.register_classes_factory(classes)
