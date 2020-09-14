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
        layout = self.layout
        s = state()
        
     
        
        col1 = layout.column(align = True)
        col1.label(text='Operator Settings')
        row = col1.row(align = True); row.prop(s, 'drawPoints')
        
        split = col1.row().split(factor=0.3)
        r1 = split.column().row(align=True); r1.prop(s, 'cHead')
        r2 = split.column().row(align=True); r2.prop(s, 'sHead')
        split = col1.row().split(factor=0.3)
        r1 = split.column().row(align=True); r1.prop(s, 'cMouth')
        r2 = split.column().row(align=True); r2.prop(s, 'sMouth')
        split = col1.row().split(factor=0.3)
        r1 = split.column().row(align=True); r1.prop(s, 'cEyes')
        r2 = split.column().row(align=True); r2.prop(s, 'sEyes')
        split = col1.row().split(factor=0.3)
        r1 = split.column().row(align=True); r1.prop(s, 'cBrows')
        r2 = split.column().row(align=True); r2.prop(s, 'sBrows')
        
        target = None
        if bpy.context.active_object:
            target = context.active_object
        checkArmature = False
        checkRigType = 'supports rigify,blenrig'
        readyToStart = False
        if target.type == 'ARMATURE':
            checkArmature = True
            if checkArmature:
                if 'rig_id' in target.data:
                    # its rigify
                    checkRigType = "RIGIFY"
                    readyToStart = True
                elif 'rig_name' in target.data:
                    # its blenrig
                    checkRigType = "BLENRIG"
                    readyToStart = True
                else:
                    readyToStart = False
                    checkRigType = 'supports rigify,blenrig'
                    pass
        checkArmature = False
        checkRigType = 'supports rigify,blenrig'
        readyToStart = False
        if target.type == 'ARMATURE':
            checkArmature = True
            if checkArmature:
                if 'rig_id' in target.data:
                    # its rigify
                    checkRigType = "RIGIFY"
                    readyToStart = True
                elif 'rig_name' in target.data:
                    # its blenrig
                    checkRigType = "BLENRIG"
                    readyToStart = True
                else:
                    readyToStart = False
                    checkRigType = 'supports rigify,blenrig'
                    pass

        col0 = layout.column(align = True)
        col0.label(text='Readyness')
        row = col0.row(align = True); row.label(text="Target: %s "%(target.name))
        row = col0.row(align = True); row.label(text="Armature: %s "%(checkArmature))
        row = col0.row(align = True); row.label(text="TYPE: %s "%(checkRigType))
        # row = col0.row(align = True); row.label(text="targetRig: %s "%(s.targetRig.name))
        #row.prop(s, 'Src_Rig', text='Rig Src', icon='ARMATURE_DATA')

        col = layout.column(align=1)
        row = col.row(align = True)
        if readyToStart:
            row.operator("wm.opencv_operator", text="Start Camera")
        else:
            row.label(text="INFO: %s "%("Select Right Thing"))








classes = (
        OpenCVAnimOperator,
        MC4B_PT_Panel)


register,unregister = bpy.utils.register_classes_factory(classes)
